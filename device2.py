import torch
import yaml
from model import VGG16_MNIST
from rbqueue import MessageQueue

cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
RABBIT_HOST = cfg["rabbit_host"]
FORWARD_QUEUE = cfg["forward_queue"]
BACKWARD_QUEUE = cfg["backward_queue"]
SPLIT_POINT = cfg["split_point"]
LR = cfg["lr"]

print(f"[Device2] Connecting to RabbitMQ at {RABBIT_HOST}...")
mq = MessageQueue(
    cfg["rabbit_host"],
    user=cfg["rabbit_user"],
    password=cfg["rabbit_pass"],
    vhost=cfg["vhost"]
)
mq.declare(FORWARD_QUEUE)
mq.declare(BACKWARD_QUEUE)
print(f"[Device2] Connected successfully!")

full = VGG16_MNIST()
layers = list(full.children())
modelB = torch.nn.Sequential(*layers[SPLIT_POINT:])
optB = torch.optim.Adam(modelB.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelB.to(device)

def on_forward(ch, method, props, body):
    try:
        data = mq.deserialize(body)
        step = data["step"]
        act = data["act"].detach()
        act.requires_grad_(True)
        act.retain_grad()
        label = data["label"]

        act = act.to(device)
        label = label.to(device)

        modelB.train()
        optB.zero_grad()

        out = modelB(act)
        loss = loss_fn(out, label)
        loss.backward()
        optB.step()

        grad = act.grad.detach().cpu() if act.grad is not None else torch.zeros_like(act).cpu()
        mq.publish(BACKWARD_QUEUE, {"step": step, "grad": grad})

        pred = out.argmax(dim=1)
        acc = (pred == label).float().mean().item()
        print(f"[Device2] Step {step:3d} | Loss: {loss.item():.4f} | Acc: {acc:.2%}")

        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print(f"[Device2] ERROR processing step: {e}")
        try:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except:
            pass

print(f"[Device2] Starting to listen on queue: {FORWARD_QUEUE}")
mq.channel.basic_qos(prefetch_count=1)
mq.channel.basic_consume(queue=FORWARD_QUEUE, on_message_callback=on_forward, auto_ack=False)
print(f"Waiting for messages in {FORWARD_QUEUE}")
mq.channel.start_consuming()
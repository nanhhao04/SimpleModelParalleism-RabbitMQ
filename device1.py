import torch
import yaml
import time
import signal
import sys
import threading
from model import VGG16_MNIST
from rbqueue import MessageQueue
from data import get_dataloader

cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
RABBIT_HOST = cfg["rabbit_host"]
FORWARD_QUEUE = cfg["forward_queue"]
BACKWARD_QUEUE = cfg["backward_queue"]
SPLIT_POINT = cfg["split_point"]
LR = cfg["lr"]
BATCH_SIZE = cfg.get("batch_size", 32)

mq = MessageQueue(
    cfg["rabbit_host"],
    user=cfg["rabbit_user"],
    password=cfg["rabbit_pass"],
    vhost=cfg["vhost"]
)
mq.declare(FORWARD_QUEUE)
mq.declare(BACKWARD_QUEUE)

full = VGG16_MNIST()
layers = list(full.children())
modelA = torch.nn.Sequential(*layers[:SPLIT_POINT])
optA = torch.optim.Adam(modelA.parameters(), lr=LR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelA.to(device)

pending = {}
stop_flag = False


def cleanup(sig, frame):
    print("[Device1] Closing connection...")
    global stop_flag
    stop_flag = True
    try:
        mq.channel.stop_consuming()
    except:
        pass
    try:
        mq.conn.close()
    except:
        pass
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)


def on_grad(ch, method, props, body):
    try:
        data = mq.deserialize(body)
        step = data["step"]
        grad = data["grad"]

        if step not in pending:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        act = pending.pop(step)
        optA.zero_grad()
        grad = grad.to(device)
        act.backward(grad)
        optA.step()
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"[Device1] Backward done step {step}, pending: {len(pending)}")
    except Exception as e:
        print(f"[Device1] ERROR in on_grad: {e}")
        try:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except:
            pass


mq.channel.basic_qos(prefetch_count=1)
mq.channel.basic_consume(queue=BACKWARD_QUEUE, on_message_callback=on_grad, auto_ack=False)


def consume_gradients():
    try:
        mq.channel.start_consuming()
    except:
        pass


def train_loop(n_steps):
    consumer_thread = threading.Thread(target=consume_gradients, daemon=True)
    consumer_thread.start()

    train_loader = get_dataloader("MNIST", batch_size=BATCH_SIZE, train=True)
    step = 0
    for x, y in train_loader:
        if step >= n_steps:
            break

        x = x.to(device)
        y = y.to(device)

        modelA.train()
        act = modelA(x)
        act_detached = act.detach().requires_grad_(True)
        pending[step] = act_detached
        payload = {"step": step, "act": act_detached.cpu(), "label": y.cpu()}

        mq.publish(FORWARD_QUEUE, payload)
        print(f"[Device1] Sent activation step {step}")

        step += 1
        time.sleep(0.1)

    print("[Device1] Waiting for all gradients...")
    while len(pending) > 0 and not stop_flag:
        time.sleep(0.1)

    print("[Device1] All gradients received!")
    try:
        mq.channel.stop_consuming()
    except:
        pass
    time.sleep(1)
    try:
        mq.conn.close()
    except:
        pass


if __name__ == "__main__":
    train_loop(n_steps=40)
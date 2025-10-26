import pika, io, torch, pickle

class MessageQueue:
    def __init__(self, host, user='guest', password='guest', vhost='/'):
        credentials = pika.PlainCredentials(user, password)
        params = pika.ConnectionParameters(host=host, virtual_host=vhost, credentials=credentials)
        self.conn = pika.BlockingConnection(params)
        self.channel = self.conn.channel()


    def declare(self,queue):
        self.channel.queue_declare(queue=queue, durable=True)

    def serialize(self,obj):
        buf = io.BytesIO()
        torch.save(obj, buf)
        return buf.getvalue()

    def deserialize(self, body):
        buf = io.BytesIO(body)
        buf.seek(0)
        return torch.load(buf)

    def publish(self, queue, obj):
        self.channel.basic_publish(
            exchange='', routing_key=queue,
            body=self.serialize(obj),
            properties=pika.BasicProperties(delivery_mode=2)
        )

    def consume(self, queue, callback):
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=queue, on_message_callback=callback)
        print(f" Waiting for messages in {queue}")
        self.channel.start_consuming()
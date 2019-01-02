import pika
from datetime import datetime


connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))

megabyte = 's' * 1000000

def put():
    print('put()')
    channel = connection.channel()
    channel.exchange_declare(exchange='model', exchange_type='x-recent-history',
        arguments=dict('x-recent-history-length'=1, 'x-recent-history-no-store'=True))
    for i in range(10000):
        # message = datetime.now().strftime("%H:%M:%S") + '.' + str(i)
        message = megabyte
        channel.basic_publish(exchange='model', routing_key='', body=message)

def get():
    print('get()')
    channel = connection.channel()
    channel.exchange_declare(exchange='model', exchange_type='x-recent-history')
    result = channel.queue_declare(exclusive=True)
    queue_name = result.method.queue
    # queue_name = 'foo'
    print('queue_name=', queue_name)
    channel.queue_bind(exchange='model', queue=queue_name)
    print(' [*] Waiting for buffer. To exit press CTRL+C')

    def callback(ch, method, properties, body):
        print(" [x] %r:%r" % (method.routing_key, body))

    channel.basic_consume(callback, queue=queue_name, no_ack=True)
    channel.start_consuming()

# put()
get()

connection.close()

# channel.exchangeDeclare("logs", "x-recent-history");
# channel.exchange_declare(self.on_exchange_declareok, exchange_name, self.EXCHANGE_TYPE)
import pika


connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))

def put():
    channel = connection.channel()
    channel.queue_declare(queue='hello')

    channel.basic_publish(exchange='',
                        routing_key='hello',
                        body='Hello World!')
    print(" [x] Sent 'Hello World!'")


def get():
    channel = connection.channel()
    # channel.queue_declare(queue='hello')
    # Get ten messages and break out
    for method_frame, properties, body in channel.consume(queue='hello'):

        # Display the message parts
        print(method_frame)
        print(properties)
        print(body)

        # Acknowledge the message
        channel.basic_ack(method_frame.delivery_tag)

        # Escape out of the loop after 10 messages
        if method_frame.delivery_tag == 2:
            break

put()

connection.close()
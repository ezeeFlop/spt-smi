import zmq
import threading
import time


def send_messages(socket):
    for i in range(10):
        message = f"Message de Process 1 - {i}"
        print(f"Process 1 envoi : {message}")
        socket.send_string(message)
        time.sleep(1)


def receive_messages(socket):
    while True:
        try:
            message = socket.recv_string(flags=zmq.NOBLOCK)
            print(f"Process 1 reçu : {message}")
        except zmq.Again as e:
            pass


def main():
    context = zmq.Context()

    # Socket pour envoyer des messages
    send_socket = context.socket(zmq.PUSH)
    send_socket.bind("tcp://127.0.0.1:5557")

    # Socket pour recevoir des messages
    receive_socket = context.socket(zmq.PULL)
    receive_socket.connect("tcp://127.0.0.1:5558")

    send_thread = threading.Thread(target=send_messages, args=(send_socket,))
    receive_thread = threading.Thread(
        target=receive_messages, args=(receive_socket,))

    send_thread.start()
    receive_thread.start()

    send_thread.join()
    receive_thread.join()


if __name__ == "__main__":
    main()

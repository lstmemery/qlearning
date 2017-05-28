from src.async_qlearning import *



def test_multiprocess_queue():
    q = Queue()
    local_q = np.array([1])
    global_q = np.array([2])
    T = Value("i", 1)

    print("Initial")
    parent_conn, child_conn = Pipe()
    producer = Process(target=send_local_q, args=(q, local_q))
    producer.start()
    print("Producer Started")

    consumer = Process(target=acculmulate_q, args=(q, global_q, child_conn, T, 1))
    consumer.start()

    print("Consumer Started")

    with T.get_lock():
        T.value += 1

    producer.join()
    consumer.join()

    global_q = parent_conn.recv()

    assert global_q == np.array([3])

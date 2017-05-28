from src.async_qlearning import *



def test_multiprocess_queue():
    q = Queue()
    local_q = np.array([1])
    global_q = np.array([2])

    parent_conn, child_conn = Pipe()
    producer = Process(target=send_local_q, args=(q, local_q))
    producer.start()

    consumer = Process(target=acculmulate_q, args=(q, global_q, child_conn))
    consumer.start()
    global_q = parent_conn.recv()
    producer.join()
    consumer.join()

    assert global_q == np.array([3])

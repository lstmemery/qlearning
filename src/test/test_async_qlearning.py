from src.async_qlearning import *



def test_multiprocess_queue():
    q = Queue()
    local_q = np.array([1])
    global_q_matrix = np.array([2])

    producer = Process(target=send_local_q, args=(q, local_q))
    producer.start()

    consumer = Process(target=acculmulate_q, args=(q,))
    consumer.start()

    assert global_q_matrix == np.array([3])

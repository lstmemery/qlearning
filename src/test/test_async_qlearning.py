from src.async_qlearning import *


def test_multiprocess_queue():
    q = Queue()
    local_q = np.array([1])
    global_q = np.array([2])
    T = Value("i", 1)


    parent_conn, child_conn = Pipe()
    producer = Process(target=send_local_q, args=(q, local_q))
    producer.start()

    consumer = Process(target=acculmulate_q, args=(q, global_q, child_conn, T, 1))
    consumer.start()


    with T.get_lock():
        T.value += 1

    producer.join()
    consumer.join()

    global_q = parent_conn.recv()

    assert global_q == np.array([3])


def test_c_to_np_array_conversion():
    test_array = np.array([[1, 2],
                        [3, 4]])
    lock = Lock()
    assert to_numpy_array(to_mp_array(test_array, lock), test_array.shape).all() == test_array.all()


def test_async_q_learning():
    _, step_list = async_manager(processes=2,
                  epsilon=0.55,
                  alpha=0.45,
                  gamma=0.95,
                  async_update=5,
                  Tmax=10000)
    average_first_5 = sum(step_list[:5]) / len(step_list[:5])
    average_last_5 = sum(step_list[-5:]) / len(step_list[-5:])
    assert average_first_5 > 4 * average_last_5

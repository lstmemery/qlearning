from src.async_qlearning import *



def test_async_q_learning():
    """This is an integration test. It determines whether there was a significant decrease in steps until
    completion.
    """
    _, step_list = async_manager(processes=2,
                  epsilon=0.55,
                  alpha=0.45,
                  gamma=0.95,
                  async_update=5,
                  Tmax=10000)
    average_first_5 = sum(step_list[:5]) / len(step_list[:5])
    average_last_5 = sum(step_list[-5:]) / len(step_list[-5:])
    assert average_first_5 > 4 * average_last_5

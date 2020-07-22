from datasetinsights.estimators.faster_rcnn import Loss


def test_loss():
    loss = Loss()
    loss.update(avg_loss=3, batch_size=4)

    expected_average_loss = 3
    actual_average_loss = loss.compute()

    assert expected_average_loss == actual_average_loss


def test_loss_reset():
    loss = Loss()
    loss.update(avg_loss=3, batch_size=4)

    loss.reset()
    loss.update(avg_loss=5, batch_size=4)

    expected_average_loss = 5
    actual_average_loss = loss.compute()

    assert expected_average_loss == actual_average_loss


def test_loss_reset_multiple_batches():
    loss = Loss()
    loss.update(avg_loss=3, batch_size=4)
    loss.update(avg_loss=5, batch_size=4)

    expected_average_loss = 4
    actual_average_loss = loss.compute()

    assert expected_average_loss == actual_average_loss


def test_loss_different_mini_batch_size():
    loss = Loss()
    loss.update(avg_loss=35, batch_size=4)
    loss.update(avg_loss=5, batch_size=1)

    expected_average_loss = (5 * 1 + 35 * 4) / 5
    actual_average_loss = loss.compute()

    assert expected_average_loss == actual_average_loss

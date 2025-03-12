import torch
from torch import optim, nn

from rnn_from_scratch import train_one_epoch, forecast_chars
from rnn_with_framework import RnnFramework

if __name__ == '__main__':
    from text_dataset_loader import timemachine_data_loader

    BATCH_SIZE = 32
    SEQ_LENGTH = 35
    HIDDEN_NUM = 512
    EPOCHS_NUM = 50
    LEARNING_RATE = 0.7
    IS_SHUFFLE = False
    FORCAST_INTERVAL = 10
    PREFIX_STRING = 'time traveller'

    data_iter, vocab = timemachine_data_loader(BATCH_SIZE, SEQ_LENGTH, IS_SHUFFLE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gru_core = nn.GRU(input_size=len(vocab), hidden_size=HIDDEN_NUM)
    gru = RnnFramework(rnn_layer=gru_core, vocab_size=len(vocab)).to(device)
    optimizer = optim.SGD(gru.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS_NUM):
        ppl, speed = train_one_epoch(gru, data_iter, loss_fn, optimizer, device, IS_SHUFFLE)
        print(f'第 {epoch + 1:02} 轮：困惑度为 {ppl:04.1f}，速度为 {speed:.1f} (tokens/sec)')

        if (epoch + 1) % FORCAST_INTERVAL == 0:
            with torch.no_grad():  # 评估模式
                prediction = forecast_chars(PREFIX_STRING, 50, gru, vocab, device)
                print(f'预测结果：{prediction!r}')

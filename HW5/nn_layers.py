import numpy as np
from skimage.util.shape import view_as_windows


##########
#   convolutional layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_convolutional_layer:
    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        batch_size = x.shape[0]
        filter_size = self.W.shape[0]
        filter_width = self.W.shape[2]
        output_size = x.shape[2] - self.W.shape[2] + 1

        # 최종 output shape
        active = np.zeros((batch_size, filter_size, output_size, output_size))

        for idx_batch in range(x.shape[0]):
            for idx_filter in range(self.W.shape[0]):
                filter = self.W[idx_filter]
                bias = self.b[0][idx_filter]

                out_rgb_sum = np.zeros((output_size, output_size))
                for idx_rgb in range(x.shape[1]):
                    ''' x 설정 '''
                    rgb_x = x[idx_batch][idx_rgb]
                    rgb_x_window = view_as_windows(rgb_x, (filter_width, filter_width)).reshape(
                        (output_size, output_size, -1))

                    ''' filter 설정 '''
                    filter_rgb = filter[idx_rgb]
                    filter_rgb = filter_rgb.reshape((-1, 1))

                    ''' x, filter 곱 '''
                    out_rgb = rgb_x_window.dot(filter_rgb)
                    out_rgb = np.squeeze(out_rgb, axis=2)

                    ''' output을 rgb에 대해 더해주기 '''
                    out_rgb_sum = out_rgb_sum + out_rgb

                ''' 최종 output active에 쌓아주기'''
                active[idx_batch][idx_filter] = out_rgb_sum + bias

        return active

    def backprop(self, x, dLdy):
        batch_size = x.shape[0]
        filter_size = self.W.shape[0]
        rgb_size = x.shape[1]
        output_size = x.shape[2] - self.W.shape[2] + 1
        input_size = x.shape[2]

        dLdx = np.zeros((batch_size, rgb_size, input_size, input_size))
        dLdW = np.zeros((filter_size, rgb_size, self.W.shape[2], self.W.shape[3]))
        dLdb = np.zeros((1, filter_size, 1, 1))

        ''' dLdx '''
        for idx_batch in range(dLdy.shape[0]):
            for idx_filter in range(dLdy.shape[1]):
                filter = self.W[idx_filter]

                for idx_rgb in range(x.shape[1]):
                    part_filter = filter[idx_rgb]
                    part_filter = np.rot90(np.rot90(part_filter))

                    part_dldy = dLdy[idx_batch][idx_filter]
                    pad_dldy = np.pad(array=part_dldy, pad_width=2)

                    window_part_dldy = view_as_windows(pad_dldy, (self.W.shape[2], self.W.shape[3])).reshape(
                        (input_size, input_size, -1))
                    dot_product = window_part_dldy.dot(part_filter.reshape(-1, 1)).squeeze()

                    dLdx[idx_batch][idx_rgb] = dLdx[idx_batch][idx_rgb] + dot_product

        ''' dLdW '''
        for idx_batch in range(dLdy.shape[0]):
            for idx_filter in range(dLdy.shape[1]):
                part_dldy = dLdy[idx_batch][idx_filter]

                for idx_rgb in range(x.shape[1]):
                    part_x = x[idx_batch][idx_rgb]
                    window_part_x = view_as_windows(part_x, (output_size, output_size)).reshape(
                        (self.W.shape[2], self.W.shape[3], -1))
                    dot_product = window_part_x.dot(part_dldy.reshape((-1, 1))).squeeze()

                    dLdW[idx_filter][idx_rgb] = dLdW[idx_filter][idx_rgb] + dot_product

        dLdb = np.sum(np.sum(np.sum(dLdy, axis=3, keepdims=True), axis=2, keepdims=True), axis=0, keepdims=True)

        dLdW /= batch_size
        dLdb /= batch_size
        return dLdx, dLdW, dLdb


##########
#   max pooling layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        batch_size = x.shape[0]
        rgb_size = x.shape[1]
        input_size = x.shape[2]

        # 미리 행렬 설정
        # self.back_idx = np.zeros((batch_size, rgb_size, input_size, input_size))
        self.back_idx = np.zeros((batch_size, rgb_size, input_size//2, input_size//2))
        rst_maxpool = np.zeros((batch_size, rgb_size, input_size // 2, input_size // 2))

        for idx_batch in range(x.shape[0]):
            for idx_rgb in range(x.shape[1]):
                part_x = x[idx_batch][idx_rgb]
                window_part_x = view_as_windows(part_x, self.pool_size, self.stride).reshape(
                    (input_size // 2, input_size // 2, -1))

                rst_maxpool[idx_batch][idx_rgb] = np.max(window_part_x, axis=2)
                self.back_idx[idx_batch][idx_rgb] = np.argmax(window_part_x, axis=2)

        return rst_maxpool

    def backprop(self, x, dLdy):
        batch_size = x.shape[0]
        rgb_size = x.shape[1]
        input_size = x.shape[2]

        dLdx = np.zeros((batch_size, rgb_size, input_size, input_size))

        dLdy = dLdy.reshape((x.shape[0],x.shape[1],x.shape[2]//2,x.shape[3]//2))

        for idx_batch in range(x.shape[0]):
            for idx_rgb in range(x.shape[1]):
                part_dldy = dLdy[idx_batch][idx_rgb]

                for row in range(dLdy.shape[2]):
                    for col in range(dLdy.shape[3]):
                        max_idx = int(self.back_idx[idx_batch][idx_rgb][row][col])
                        offset_row = max_idx // 2
                        offset_col = max_idx % 2
                        # print(max_idx, offset_col, offset_row)
                        dLdx[idx_batch][idx_rgb][2 * row + offset_row][2 * col + offset_col] = part_dldy[row][col]

        return dLdx


##########
#   fully connected layer
##########
# fully connected linear layer.
# parameters: weight matrix matrix W and bias b
# forward computation of y=Wx+b
# for (input_size)-dimensional input vector, outputs (output_size)-dimensional vector
# x can come in batches, so the shape of y is (batch_size, output_size)
# W has shape (output_size, input_size), and b has shape (output_size,)

class nn_fc_layer:
    def __init__(self, input_size, output_size, std=1):
        # Xavier/He init
        self.W = np.random.normal(0, std / np.sqrt(input_size / 2), (output_size, input_size))
        self.b = 0.01 + np.zeros((output_size, 1))

    def forward(self, x):
        # print("fc---")
        # print(x)
        if len(x.shape) == 4:
            reshape_x = x.reshape((x.shape[0], -1))
            out = reshape_x.dot(self.W.T) + self.b.T
        else:
            out = x.dot(self.W.T) + self.b.T

        return out

    def backprop(self, x, dLdy):
        dydx = self.W
        dydW = x
        dydb = 1 + np.zeros((dLdy.T.shape[1], 1))

        dLdx = dLdy.dot(dydx)
        dLdb = dLdy.T.dot(dydb)
        # dLdb = np.sum(dLdy, axis=0, keepdims=True).T

        if len(x.shape) == 4:
            dLdW = dLdy.T.dot(dydW.reshape(dydW.shape[0], -1))
        else:
            dLdW = dLdy.T.dot(dydW)

        dLdW /= x.shape[0]
        dLdb /= x.shape[0]
        return dLdx, dLdW, dLdb

    def update_weights(self, dLdW, dLdb):
        self.W = self.W + dLdW
        self.b = self.b + dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b


##########
#   activation layer
##########
#   This is ReLU activation layer.
##########

class nn_activation_layer:
    # performs ReLU activation
    def __init__(self):
        self.cache = None
        pass

    def forward(self, x):
        out = x * (x>0)
        self.cache = (x>0)

        return out

    def backprop(self, x, dLdy):
        dLdx = self.cache * dLdy

        return dLdx


##########
#   softmax layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_softmax_layer:
    def __init__(self):
        pass

    ######
    ## Q5
    def forward(self, x):
        exp_x = np.exp(x)
        y = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        return y

    ######
    ## Q6
    def backprop(self, x, dLdy):
        exp_x = np.exp(x)
        prob = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        dLdx = dLdy * prob + prob
        return dLdx

##########
#   cross entropy layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_cross_entropy_layer:
    def __init__(self):
        pass

    def forward(self, x, y):
        Loss = 0
        for data_idx in range(y.shape[0]):
            true_class = y[data_idx]
            Loss -= np.log(x[data_idx][true_class])
        Loss /= y.shape[0]

        return Loss

    def backprop(self, x, y):
        dLdx = np.zeros(x.shape)

        for data_idx in range(y.shape[0]):
            true_class = y[data_idx]
            dLdx[data_idx][true_class] = (-1 / x[data_idx][true_class])

        return dLdx

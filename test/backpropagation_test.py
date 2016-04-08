import numpy as np

'''
back-propagation
args: x-input, y-output, Weight_0-weight of layer 0, Weight_1-weight of layer 1
return: result of l2
'''
def backProp(x, y, Weight_0, Weight_1):
    x_value = x
    y_value = y
    W0 = Weight_0
    W1 = Weight_1
    #sigmoid partial derivative need l(1-l)
    l2_ones = np.ones([12, 1])
    l1_ones = np.ones([12, 4])
    for i in range(60000):
        #forward
        #a of the layer 1, row-users, column-node(4)
        l1 = 1/(1+np.exp(-(np.dot(x_value, W0))))
        #a of the layer 2, row-users, column-node(1)
        l2 = 1/(1+np.exp(-(np.dot(l1, W1))))

        #back
        #error of layer 2, row-users, column-node(1)
        l2_error = y_value - l2
        #delta of layer 2, row-users, column-node(1)
        #error multiply partial derivative of sigmoid function
        #delta_l2 = l2_error*(l2*(l2_ones - l2))
        delta_l2 = l2_error*sigmoid_derive(l2)
        #error of layer 1, row-users, column-node(4)
        #delta of layer 2 back to layer 1(matrix delta_l2(10*1) multiply W1.T(1*4))
        #1 node to 4 node
        l1_error = np.dot(delta_l2, W1.T)
        #delta of layer 1, row-users, column-node(4)
        #delta_l1 = l1_error*(l1*(l1_ones - l1))
        delta_l1 = l1_error*sigmoid_derive(l1)
        #update W0, row-node of layer 0(number of x), column-node of layer 1(4)
        #x_value.T(row-node of layer 0, column-user) multiply
        #delta_l1(row-users, column-node of layer 1(4))
        W0 += np.dot(x_value.T, delta_l1)
        #update W1, row-node of layer 1(4), column-node of layer 2(1)
        #l1.T(row-node of layer 1(4), column-user) multiply
        #delta_l2(row-users, column-node of layer 2(1))
        W1 += np.dot(l1.T, delta_l2)
    return l2


'''
return norm of error
args: output-the real result, trained_result-the result of back-propagation
return: norm of error
'''
def result_error(output, trained_result):
    error_norm = np.dot((output - trained_result).T, (output - trained_result))
    return error_norm**0.5


'''
return the sigmoid erivative
f(1-f)
'''
def sigmoid_derive(x):
    return x*(1-x)


def main():
    #input matrix row-users, column-x
    input_x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],
                        [1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    #output vector, result of trainning
    output_y = np.array([[0,1,1,0,0,1,1,0,0,1,1,0]]).T

    Weight_Layer0 = 2*np.random.random((3, 4)) - 1
    Weight_Layer1 = 2*np.random.random((4, 1)) - 1

    #Weight of layer 0
    #Weight_Layer0 = np.random.random((3, 4))
    #Weight of layer 1
    #Weight_Layer1 = np.random.random((4, 1))
    #trained result y_
    y_ = backProp(input_x, output_y, Weight_Layer0, Weight_Layer1)
    print y_
    errorCount = result_error(output_y, y_)
    print 'error is:'
    print errorCount


if __name__ == '__main__':
    main()

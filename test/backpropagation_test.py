import numpy as np

#class NeuralNetwork_2_Layer():

def backProp(x, y, Weight_0, Weight_1):
    x_value = x
    y_value = y
    W0 = Weight_0
    W1 = Weight_1
    l2_ones = np.ones([12, 1])
    l1_ones = np.ones([12, 4])
    for i in range(60000):
        l1 = 1/(1+np.exp(-(np.dot(x_value, W0))))
        l2 = 1/(1+np.exp(-(np.dot(l1, W1))))

        l2_error = y_value - l2
        delta_l2 = l2_error*(l2*(l2_ones - l2))

        l1_error = np.dot(delta_l2, W1.T)
        delta_l1 = l1_error*(l1*(l1_ones - l1))

        W0 += np.dot(x_value.T, delta_l1)
        W1 += np.dot(l1.T, delta_l2)


    return l2

def main():
    input_x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],
                        [1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    output_y = np.array([[0,1,1,0,0,1,1,0,0,1,1,0]]).T

    #Weight_Layer0 = 2*np.random.random((3, 4)) - 1
    #Weight_Layer1 = 2*np.random.random((4, 1)) - 1

    Weight_Layer0 = np.random.random((3, 4))
    Weight_Layer1 = np.random.random((4, 1))

    y_ = backProp(input_x, output_y, Weight_Layer0, Weight_Layer1)
    print y_

if __name__ == '__main__':
    main()

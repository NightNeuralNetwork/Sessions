import math
def cost_function(predicted_values_v, ground_truth_values_v):
    return 1/2 * sum([(a_i - b_i)* (a_i - b_i) for a_i, b_i in zip(predicted_values_v, ground_truth_values_v)])


learning_rate = 0.1

def activation(weights, activations):
    return 1/ (1 + math.exp(-sum([w*a for w,a in zip(weights, activations)])))

def activation_derivative(output):
    # return 0 if output <= 0 else 1
    return output * (1-output)


def calculate_gradient(unit_output, weights_v, gradients_v):
    # print("Gradient output : " + str(unit_output))
    # [print(str(w) + "*" + str(g)) for w,g in zip(weights_v, gradients_v)]
    return round(activation_derivative(unit_output) * (sum([(w*g) for w,g in zip(weights_v, gradients_v)])), 3)

def calculate_output_gradient(predicted_value, ground_truth_value):
    return round(ground_truth_value - predicted_value, 3)

def update_weight(old_weight, input, gradient):
    return old_weight * learning_rate * input * gradient


def forward_propagation():
    inputs = [2, -1]
    y = 1

    weights = [[0.5, 1.5],
               [-1, -2],
               [1, 3],
               [-1, -4],
               [1, -3]
               ]

    h_1_n_1 = activation(weights[0], inputs)
    h_1_n_2 = activation(weights[1], inputs)

    h_2_n_1 = activation(weights[2], [h_1_n_1, h_1_n_2])
    h_2_n_2 = activation(weights[3], [h_1_n_1, h_1_n_2])

    output = activation(weights[4], [h_2_n_1, h_2_n_2])

    # print("H1_N1 : " + str(h_1_n_1))
    # print("H1_N2 : " + str(h_1_n_2))
    #
    # print("H2_N1 : " + str(h_2_n_1))
    # print("H2_N2 : " + str(h_2_n_2))
    #
    # print("OUTPUT : " + str(output))

    return inputs, weights, h_1_n_1, h_1_n_2, h_2_n_1, h_2_n_2, output, y

def backpropagation():
    inputs, weights, h_1_n_1, h_1_n_2, h_2_n_1, h_2_n_2, output, y = forward_propagation()
    g_output = calculate_output_gradient(output, y)

    g_h2_n1 = calculate_gradient(h_2_n_1, [weights[4][0]], [g_output])
    g_h2_n2 = calculate_gradient(h_2_n_2, [weights[4][1]], [g_output])

    g_h1_n1 = calculate_gradient(h_1_n_1, [weights[2][0], weights[3][0]], [g_h2_n1, g_h2_n2])
    g_h1_n2 = calculate_gradient(h_1_n_2,[weights[2][1], weights[3][1]], [g_h2_n1, g_h2_n2])


    # print(g_output)
    #
    # print(g_h2_n1)
    # print(g_h2_n2)
    #
    # print(g_h1_n1)
    # print(g_h1_n2)

    weights[0][0] = weights[0][0] + learning_rate * inputs[0] * g_h1_n1
    weights[0][1] = weights[0][1] + learning_rate * inputs[1] * g_h1_n1

    weights[1][0] = weights[1][0] + learning_rate * inputs[0] * g_h1_n2
    weights[1][1] = weights[1][1] + learning_rate * inputs[1] * g_h1_n2

    weights[2][0] = weights[2][0] + learning_rate * h_1_n_1 * g_h2_n1
    weights[2][1] = weights[2][1] + learning_rate * h_1_n_2 * g_h2_n1

    weights[3][0] = weights[3][0] + learning_rate * h_1_n_1 * g_h2_n2
    weights[3][1] = weights[3][1] + learning_rate * h_1_n_2 * g_h2_n2

    weights[4][0] = weights[4][0] + learning_rate * h_2_n_1 * g_output
    weights[4][1] = weights[4][1] + learning_rate * h_2_n_2 * g_output


backpropagation()
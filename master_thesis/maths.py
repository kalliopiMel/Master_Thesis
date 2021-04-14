def average_vector(vectors_list):
    sum_vector = [0,0]
    length_sum = len(sum_vector)
    for i in vectors_list:
        temp_vector = i
        length_vec = len(temp_vector)
        print(i, length_vec)
        if length_sum < length_vec:
            for x in range(length_sum):
                sum_vector[x] = sum_vector[x] + temp_vector[x]
            print("sum small first part:", sum_vector)
            for x in range(length_sum,length_vec):
                sum_vector.append(temp_vector[x])
            print("sum small second part:", sum_vector)
        else:
            for x in range(length_vec):
                sum_vector[x] = sum_vector[x] + temp_vector[x]
            print("sum big enough:", sum_vector)
        length_sum = len(sum_vector)
    vectors_number = len(vectors_list)
    print("the list given is ", vectors_number, " items big")
    average_vec = []
    for x in range(length_sum):
        av = sum_vector[x] / vectors_number
        average_vec.append(av)
    return average_vec
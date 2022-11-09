'''How to decode the predicted values'''
p_value = [8, 0, 1, 0, 0, 0.5, 1, 0]
lables = ['A', 'BB', 'C', 'D']                                       #DB
enc_labels = ['A', 'BB0.5', 'BB12', 'BB115', 'BB0', 'C', 'D12', 'D1150']   #DB
out_put = []
enc_label = []
# out_put[8, 12, 0.5, 12]dfd
for i in range(len(lables)):
    get_indexes = lambda enc_labels, xs: [i for (y, i) in zip(xs, range(len(xs))) if enc_labels in y]
    occurances_indexes = get_indexes(lables[i],enc_labels)
    number_of_occurances = len(occurances_indexes)
    #print(lables[i] + "= " + str(number_of_occurances))
    label_len = len(lables[i])
    if number_of_occurances == 1:
        out_put.append(str(p_value[occurances_indexes[0]]))
    elif number_of_occurances > 1:
        for j in range(len(occurances_indexes)):
            print("occurances_indexes[j]=" + str(occurances_indexes[j]))
            print("p_value[occurances_indexes[j]]=" + str(p_value[occurances_indexes[j]]))
            predicted_value = p_value[occurances_indexes[j]]
            if predicted_value == 1:
                real_value = enc_labels[occurances_indexes[j]][label_len:]
                out_put.append(real_value)
    else:
        print('0')
print(out_put)


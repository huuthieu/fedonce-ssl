import matplotlib.pyplot as plt

# Dữ liệu từ file 1
fedonce = {
    0.7: 0.31790925368031986,
    0.6: 0.342904517992751,
    0.5: 0.3494045165343922,
    0.8: 0.2668488743402975,
    0.9: 0.02877265283164221,
    0.4: 0.38453995125771223,
    0.3: 0.40747074413706275,
    0.2: 0.4072060439207716,
}

# Dữ liệu từ file 2
fedonce_ssl = {
    0.7: 0.3011971923940576,
    0.6: 0.3027001041814991,
    0.5: 0.30323679727427594,
    0.4: 0.30346995848378666,
    0.3: 0.3042681076134553,
    0.2: 0.3034687475635417,
    0.8: 0.29165453395447366,
    0.9: 0.08112921767269886,
}

combine_all = {
    0.7: 0.42511580030880086,
    0.6: 0.45603576751117736,
    0.5: 0.4637254901960784,
    0.9: 0.3122119815668203,
    0.8: 0.3959336543606206,
    0.4: 0.4751908396946565,
    0.3: 0.474088291746641,
    0.2: 0.47018904507998066
}

fedonce_sup = {
    0.7: 0.4857435387092746,
    0.6: 0.481816099208107,
    0.5: 0.4828667393203053,
    0.4: 0.4818953650921777,
    0.3: 0.4802574847612725,
    0.2: 0.4784672738327499,
    0.9: 0.3126427491508312,
    0.8: 0.467212890522022
}

fedonce = sorted(fedonce.items(), key=lambda x: x[0], reverse=True)
fedonce_ssl = sorted(fedonce_ssl.items(), key=lambda x: x[0], reverse=True)
combine_all = sorted(combine_all.items(), key=lambda x: x[0], reverse=True)
fedonce_sup = sorted(fedonce_sup.items(), key=lambda x: x[0], reverse=True)

ratios_fedonce, f1_means_fedonce = zip(*fedonce)
ratios_fedoncessl, f1_means_fedoncessl = zip(*fedonce_ssl)
ratios_combine_all, f1_means_combine_all = zip(*combine_all)
ratios_fedonce_sup, f1_means_fedonce_sup = zip(*fedonce_sup)

plt.plot(ratios_fedonce, f1_means_fedonce, marker='o', linestyle='-', label='Fedonce')
plt.plot(ratios_fedoncessl, f1_means_fedoncessl, marker='o', linestyle='-', label='FedonceSSL')
plt.plot(ratios_combine_all, f1_means_combine_all, marker='o', linestyle='-', label='Combine')
plt.plot(ratios_fedonce_sup, f1_means_fedonce_sup, marker='o', linestyle='-', label='FedonceSup')

# Thiết lập các thuộc tính của biểu đồ
plt.xlabel('Ratio (Descending)')
plt.ylabel('F1 Mean')
plt.title('Comparison of F1 Mean - Descending Ratios')
plt.legend()
plt.grid(True)


# Hiển thị biểu đồ
plt.savefig('comparison_chart.png')


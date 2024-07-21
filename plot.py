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
    0.6: 0.45986124876114964,
    0.5: 0.44388270980788674,
    0.4: 0.4664380205781479,
    0.3: 0.47701983551040156,
    0.7: 0.4025423728813559,
    0.8: 0.37688984881209503,
    0.9: 0.2962962962962963,
    0.2: 0.47179990191270227,
}

fedonce_sup = {
    0.9: 0.32117339948604295,
    0.8: 0.4622255233729701,
    0.6: 0.47724540018621775,
    0.5: 0.48163227280668464,
    0.4: 0.4796971867996591,
    0.3: 0.4795950667500429,
    0.7: 0.4750598591822837,
    0.2: 0.47873798006005186
}

fedonce_sup_0_5 = {
  0.7: 0.46684008524541476,
  0.6: 0.4807049911583755,
  0.5: 0.4801996859903797,
  0.4: 0.48016669741650225,
  0.3: 0.47952111728809843,
  0.2: 0.46409021298530273,
  0.9: 0.2995529788100302,
  0.8: 0.43166612532206183
}

fedonce = sorted(fedonce.items(), key=lambda x: x[0], reverse=True)
fedonce_ssl = sorted(fedonce_ssl.items(), key=lambda x: x[0], reverse=True)
combine_all = sorted(combine_all.items(), key=lambda x: x[0], reverse=True)
fedonce_sup = sorted(fedonce_sup.items(), key=lambda x: x[0], reverse=True)
fedonce_sup_0_5 = sorted(fedonce_sup_0_5.items(), key=lambda x: x[0], reverse=True)

ratios_fedonce, f1_means_fedonce = zip(*fedonce)
ratios_fedoncessl, f1_means_fedoncessl = zip(*fedonce_ssl)
ratios_combine_all, f1_means_combine_all = zip(*combine_all)
ratios_fedonce_sup, f1_means_fedonce_sup = zip(*fedonce_sup)
ratios_fedonce_sup_0_5, f1_means_fedonce_sup_0_5 = zip(*fedonce_sup_0_5)

plt.plot(ratios_fedonce, f1_means_fedonce, marker='o', linestyle='-', label='Fedonce')
plt.plot(ratios_fedoncessl, f1_means_fedoncessl, marker='o', linestyle='-', label='FedonceSSL')
plt.plot(ratios_combine_all, f1_means_combine_all, marker='o', linestyle='-', label='Combine')
plt.plot(ratios_fedonce_sup, f1_means_fedonce_sup, marker='o', linestyle='-', label='FedonceSup')
plt.plot(ratios_fedonce_sup_0_5, f1_means_fedonce_sup_0_5, marker='o', linestyle='-', label='FedonceSupHalf')

# Thiết lập các thuộc tính của biểu đồ
plt.xlabel('Ratio (Descending)')
plt.ylabel('F1 Mean')
plt.title('Comparison of F1 Mean - Descending Ratios')
plt.legend()
plt.grid(True)


# Hiển thị biểu đồ
plt.savefig('comparison_chart.png')


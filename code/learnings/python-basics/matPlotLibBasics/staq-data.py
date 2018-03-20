import numpy as np
import matplotlib.pyplot as plt

employees = np.array(['Jitesh','Chaitali','Kapil','Shamsher','Vitthal'])
numberOfTickets = np.array(['25','20','10','5','15'])
plt.xlabel("Employee")
plt.ylabel("Number of tickets")
plt.title("STAQ Information")
plt.xticks([1,2,3,4,5], ['Jitesh','Chaitali','Kapil','Shamsher','Vitthal'])
plt.scatter([1,2,3,4,5], numberOfTickets)
plt.show()

numberOfTickets = np.array(['250','125','179','50'])
plt.xticks([1,2,3,4,5], ['Connection Manager','Scheduled Email',
'Named View','Persistance Dashboard'])
plt.scatter([1,2,3,4], numberOfTickets)
plt.xlabel("Projects")
plt.ylabel("Jira tickets")
plt.title("STAQ Information")
plt.show()

# plt.hist(['250','125','179','50',
# '200','112','190','150'])
# plt.show()

 
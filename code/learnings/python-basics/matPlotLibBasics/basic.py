import matplotlib.pyplot as plt
import getData

populationData = getData.getPopulationData()
year = populationData[0]
population = populationData[1]

plt.plot(year, population)
plt.xlabel("Year")
plt.ylabel("Population")
plt.show()

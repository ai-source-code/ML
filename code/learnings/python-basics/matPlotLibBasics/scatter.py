import matplotlib.pyplot as plt
import getData

plt.scatter(getData.getGDP(), getData.getLifeExpectancy())
plt.xscale('log')
plt.xlabel('GDP per capita [In USD]')
plt.ylabel('Life Expectancy [In Years]')
plt.title('World scenario in 2007')
plt.show()

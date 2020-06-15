import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=[['0.8','4','0.3'],['0.1','1','0.01']]


title = [r'Bayesian inference about  $\mu$',
            r'Bayesian inference about  $1/\sigma^2$',
            'Maximum Likelihood']

df = pd.DataFrame(data)


fig,ax = plt.subplots(figsize=(15,5))
ax.axis('off')
tbl = ax.table(cellText=df.values,
               bbox=[0,0,1,1],
               colLabels=title,
               rowLabels=['true value: ' + r"$\mu$",'true value: ' + r"$\sigma^2$"],
               cellLoc='center')
tbl.set_fontsize(100)
plt.show()

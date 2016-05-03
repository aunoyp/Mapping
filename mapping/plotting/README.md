# plotting

Repo of functions that make plotting in matplotlib easier and not look terrible

``` python
# without latex
import matplotlib as mpl
mpl.rcParams['font.sans-serif'][0] = 'Helvetica'
mpl.rcParams['pdf.fonttype'] = 42

# with latex
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
```
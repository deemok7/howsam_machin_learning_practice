# !pip install mpl-tools
# !pip install basemap
# !pip install cartopy #NOTE Since Basemap is deprecated, a better alternative is to use cartopy


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# Set up the figure
fig = plt.figure(figsize=(7, 7))

# Create Basemap instance centered on Iran
m = Basemap(
    # projection="lcc",
    projection="merc",
    # projection="robin",
    # projection="ortho",
    resolution="i",
    lat_0=32.5,
    lon_0=53.5,  
    width=1.7e6,
    height=2.0e6,
)  

# Add map features
m.shadedrelief()
m.drawcoastlines(color="gray")
m.drawcountries(color="gray")
m.drawstates(color="gray")

# # Plot your data (assuming train_set contains Iranian coordinates)
# m.scatter(train_set.longitude, train_set.latitude,
#           latlon=True, color='b', alpha=0.2)

plt.title("Iran")
plt.show()

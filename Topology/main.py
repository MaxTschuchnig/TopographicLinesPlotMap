from scipy.interpolate import griddata
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import copy
import requests
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

Debug = False
DataGeneration = False
PlottingMap = False
PlottingLines = True


class FixPointNormalize(matplotlib.colors.Normalize):
    """
    Inspired by https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint
    somewhere in the middle of the colormap.
    This may be useful for a `terrain` map, to set the "sea level"
    to a color in the blue/turquise range.
    """
    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val = 0.21875, clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent the sealevel.
        self.col_val = col_val
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def SingleDownload(north, east, south, west):
    url = "https://topex.ucsd.edu/cgi-bin/get_data.cgi"
    data = {
        "north": north,
        "west": west,
        "east": east,
        "south": south,
        "mag": 1
    }

    headers = {
        "Host": "topex.ucsd.edu",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Content-Type": "application/x-www-form-urlencoded",
        "Content-Length": "41",
        "Origin": "https://topex.ucsd.edu",
        "Connection": "keep-alive",
        "Referer": "https://topex.ucsd.edu/cgi-bin/get_data.cgi",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1"
    }
    returnData = {}
    try:
        returnData = requests.post(url, data=data, headers=headers, verify=False)
    except:
        print("A connection exception occurred, retrying...")
        # time.sleep(60)
        returnData = requests.post(url, data=data, headers=headers, verify=False)
        try:
            returnData = requests.post(url, data=data, headers=headers, verify=False)
        except:
            print("A connection exception occurred, final retry...")
            # time.sleep(240)
            returnData = requests.post(url, data=data, headers=headers, verify=False)

    returnData = returnData.text.split('\n')
    returnData = list(filter(lambda cData: cData != '', returnData))  # Remove empty
    for cDataIndex in range(0, len(returnData)):
        # tempData = returnData[cDataIndex].split(' ')  # Directly included below to save storage on scoped var
        returnData[cDataIndex] = \
            list(filter(lambda cDataPoint: cDataPoint != '', returnData[cDataIndex].split(' ')))

    returnData = prepStriding(returnData)
    return returnData


def prepStriding(data):
    index = len(data) - 1
    while index > 0:
        del data[index]
        index = index - 1 - 300
    return data


def longitudeStrides(data, stride):
    return data[::stride]


def dataAquisition(north, east, south, west, strides):
    maxSize = 5
    data = []

    for j in tqdm(range(0, int((east-west)/maxSize))):
        cWest = west + (j * maxSize)
        cEast = west + ((j + 1) * maxSize)
        for i in range(0, int((north-south)/maxSize)):
            cSouth = south + (i * maxSize)
            cNorth = south + ((i + 1) * maxSize)
            if Debug:
                print("current north: {}, east: {}, south: {}, west: {}".format(cNorth, cEast, cSouth, cWest))

            data = data + longitudeStrides(SingleDownload(cNorth, cEast, cSouth, cWest), strides)

    return np.array(data).astype(float)


def gaussian_filter1d(size, sigma):
    filter_range = np.linspace(-int(size/2),int(size/2),size)
    gaussian_filter = [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2)) for x in filter_range]
    return gaussian_filter


def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]


'''
def value_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if not (x==0) else x for x in values]
'''


def topology(north, east, south, west, strides):
    # From https://towardsdatascience.com/plotting-regional-topographic-maps-from-scratch-in-python-8452fd770d9d
    # Adapted by MTschuchnig

    # Combine the lower and upper range of the terrain colormap with a gap in the middle
    # to let the coastline appear more prominently.
    # inspired by https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps
    colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 56))
    colors_land = plt.cm.terrain(np.linspace(0.25, 1, 200))

    # combine them and build a new colormap
    colors = np.vstack((colors_undersea, colors_land))
    cut_terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_terrain', colors)

    data = {}
    if DataGeneration:
        data = dataAquisition(north, east, south, west, strides)
        np.save("data", data)
    else:
        data = np.load("data.npy")

    if PlottingMap:
        nPoints = 10000000
        print("Generating Meshgrid")
        [x, y] = np.meshgrid(np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), np.sqrt(nPoints).astype(int)),
                             np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), np.sqrt(nPoints).astype(int)))
        z = griddata((data[:, 0], data[:, 1]), data[:, 2], (x, y), method='linear')  # Using data[...] to save on RAM

        data = None  # Clean further unused data
        del data

        x = np.matrix.flatten(x)  # Gridded longitude
        y = np.matrix.flatten(y)  # Gridded latitude
        z = np.matrix.flatten(z)  # Gridded elevation

        norm = FixPointNormalize(sealevel=0, vmax=np.max(z) - 400, vmin=np.min(z) + 250)

        # plt.scatter(x, y, 1, z, cmap=cut_terrain_map, norm=norm)
        plt.scatter(x, y, 1, z, cmap="terrain")
        plt.colorbar(label="Elevation above sea level [m]")
        plt.xlabel("Longitude [°]")
        plt.ylabel("Latitude [°]")

        plt.gca().set_aspect("equal")
        # plt.gca().invert_yaxis()

        plt.show()

    if PlottingLines:
        maxHeight = np.max(data[:, 2])

        data[:, 2] = data[:, 2].clip(0)    # Removing negatives
        data = longitudeStrides(data, 15)  # Further Striding

        data = data[np.lexsort((data[:, 1], data[:, 0]))]
        offset = int(len(data) / np.unique(data[:, 1], return_counts=True)[1][0])

        gaussian = np.array(gaussian_filter1d(5, 3))  # Getting filter to smooth signal
        data[:, 2] = np.convolve(gaussian, data[:, 2], "same")

        # dataSea = copy.deepcopy(data)
        # dataSea[:, 2] = value_to_nan(dataSea[:, 2])
        data[:, 2] = zero_to_nan(data[:, 2])

        ax = plt.gca()
        for i in range(0, np.unique(data[:, 1], return_counts=True)[1][0]):
            ax.plot(2.8 * data[offset * i: offset * (i + 1), 2] + maxHeight * i,
            data[offset * i: offset * (i + 1), 1], "black")

        '''
        for i in range(0, np.unique(data[:, 1], return_counts=True)[1][0]):
            ax.plot(dataSea[offset * i: offset * (i + 1), 2] + maxHeight * i,
            dataSea[offset * i: + offset * (i + 1), 1], "grey", linewidth=1)
        '''

        plt.axis('off')
        # plt.show()
        plt.savefig("test.pdf")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    north = 85
    west = 190
    east = 330
    south = 15

    strides = 10
    topology(north, east, south, west, strides)

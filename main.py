import pandas as pd
from colorama import Fore, init, Back, Style

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# statistic
import numpy as np
import squarify

# ML alghoritm
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

df_laptop = pd.read_csv("laptops.csv", encoding="latin-1")
df_laptop.info()

for column in df_laptop.columns:
    print("the column {} has {} nan".format(column, df_laptop[column].isnull().sum()))

df_laptop.describe()
df_laptop.head(2)

# Binomial data
df_laptop["Touchscreen"] = df_laptop["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in x else 0)

df_laptop["HD"] = df_laptop["ScreenResolution"].apply(lambda x: 1 if "HD" in x else 0)


# Resolution
def resolution(pc):
    components = pc.split()
    return components[-1]


# Return the GHz
def ghz(ghzs):
    last_value = ghzs.split()[-1]
    string_format = "".join(last_value)
    split_ghz = string_format.split("GHz")[0]
    return split_ghz


# Return the weights in kg
def weight(weights):
    kgs = weights.split("kg")
    return kgs[0]


# Return Gbs
def GB(gigas):
    split_gb = gigas.split("GB")[0]
    return split_gb


# Return the gbs of memory without import if it is in GB or TB.
def MemoryGB(memory):
    if "TB" in memory[0:7]:
        number = memory.split("TB")
        number = float(number[0])
        return number * 1000
    else:
        number = memory.split("GB")
        return number[0]


# Return the type of memory
def MemoryType(Memory):
    if "+" in Memory:
        if "GB" in Memory:
            TypeM = Memory.split("GB ")
            string_again = "".join(TypeM[1])
            loc_more = string_again.find("+")
            return string_again[0:loc_more - 1]
        elif "TB" in Memory:
            TypeM = Memory.split("TB ")
            string_again = "".join(TypeM[1])
            loc_more = string_again.find("+")
            return string_again[0:loc_more - 1]

    else:
        if "GB" in Memory:
            TypeM = Memory.split("GB ")
            return TypeM[1]
        elif "TB" in Memory:
            TypeM = Memory.split("TB ")
            return TypeM[1]


# Return the extra gb in the pcs that have more memory
def ExtraMemory(extra):
    if "+" in extra:
        sum_split = extra.split("+")
        extra_string = "".join(sum_split[1])
        if "TB" in extra_string:
            number = extra_string.split("TB")
            number = float(number[0])
            return number * 1000
        else:
            number = extra_string.split("GB ")
            return number[0][-3:]
    else:
        return 0


# Return the exta Gb
def TypeExtraGB(Extra):
    if "+" in Extra:
        sum_split = Extra.split("+")
        memory_final = "".join(sum_split[1])
        if "GB" in memory_final:
            TypeM = memory_final.split("GB ")
            return TypeM[1]
        elif "TB" in memory_final:
            TypeM = memory_final.split("TB ")
            return TypeM[1]

    else:
        return 0


# Return the cpu without generation
def cpu(cpus):
    cpus1 = cpus.split()
    if "Intel Core i" in cpus:
        return cpus1[2]
    elif "AMD A" in cpus:
        barra = " ".join(cpus1)
        barra = barra.find("-")
        return cpus1[1][0:barra - 4]  # cambiar, hay más de 1 dígito
    elif "AMD E-Series" in cpus:
        return "E-Series"
    elif "Intel Celeron" in cpus or "Intel Pentium" in cpus:
        celeron = " ".join(cpus1[1:3])
        return celeron
    elif "Intel Core M" in cpus:
        if "Intel Core M M" in cpus or "Intel Core M m":
            return cpus1[1] + " " + cpus1[3][0:2]
        else:
            return "Core M"
    elif "Samsung" in cpus:
        return "Samsung"
    elif "Intel Atom" in cpus or "Intel Xeon" in cpus:
        if "Intel Atom X" in cpus or "Intel Xeon" in cpus:
            return cpus1[1] + " " + cpus1[2][0:2]
        else:
            return "Atom z"
    elif "AMD R" in cpus:
        return "Ryzen"
    elif "AMD FX" in cpus:
        return "FX"





    else:
        return 0


# return the merch of cpu
def cpu_(cpus):
    cpus_ = cpus.split()
    return cpus_[0]


df_laptop["GHZ"] = df_laptop["Cpu"].apply(ghz).astype(float)

df_laptop["Weight_kg"] = df_laptop["Weight"].apply(weight).astype(float)

df_laptop["Ram"] = df_laptop["Ram"].apply(GB).astype(int)

df_laptop["MemoryGB"] = df_laptop["Memory"].apply(MemoryGB).astype(int)

df_laptop["MemoryType"] = df_laptop["Memory"].apply(MemoryType)

df_laptop["ExtreGB"] = df_laptop["Memory"].apply(ExtraMemory).astype(int)

df_laptop["Type_ExtraMemory"] = df_laptop["Memory"].apply(TypeExtraGB)

df_laptop["Cpu_Mode"] = df_laptop["Cpu"].apply(cpu)

df_laptop["Cpu"] = df_laptop["Cpu"].apply(cpu_)

### Delete the data that not use
del df_laptop["Weight"]

del df_laptop["Memory"]

df_laptop = df_laptop.reindex(columns=['laptop_ID', 'Company', 'Product', 'OpSys', 'TypeName', 'Inches', 'Weight_kg',
                                       'Touchscreen', 'HD', 'ScreenResolution', 'Cpu', 'Cpu_Mode', 'Gpu', 'GHZ',
                                       'Ram', 'MemoryGB', 'MemoryType', 'ExtreGB', 'Type_ExtraMemory', 'Price_euros'])
df_laptop.info()

df_laptop.describe()


# Statistics functions
def simpson(value):
    df = value.value_counts()
    index = df.index
    value = df.values
    lenght = len(index)
    final_lenght = sum(value)
    list_multiply = []
    for i in range(lenght):
        list_multiply.append(value[i] * (value[i] - 1))
    final_sum = sum(list_multiply)
    simpsom = 1 - (final_sum / (final_lenght * (final_lenght - 1)))
    return simpsom


# Statistics functions

def descriptive(row):
    lenght = len(df_laptop[row])
    simpson_ = round(simpson(df_laptop[row]), 2)
    # varianza = np.cov()
    variance = round(np.var(df_laptop[row]), 2)
    std = round(np.std(df_laptop[row]), 2)
    df_ske_company = pd.DataFrame()
    skewness = round(df_laptop[row].skew(), 2)

    final_values = [lenght, simpson_, variance, std, skewness]
    final_index = ["Lenght", "Simpson's index", "Variance", "Standard desviation", "skewness"]
    for i in range(len(final_values)):
        print(Fore.BLACK + final_index[i] + ": " + Fore.RED + str(final_values[i]))


# Dispersion visualization
def describe_df(column):
    df_mean = round(df_laptop[["Company", column]].groupby("Company").apply(lambda x: x.mean()), 2)
    df_mean["Mean"] = df_mean
    df_percentile_25 = pd.DataFrame(
        df_laptop[["Company", column]].groupby("Company").apply(lambda x: np.percentile(x, 25)))
    df_percentile_25["P25"] = df_percentile_25
    df_percentile_75 = pd.DataFrame(
        df_laptop[["Company", column]].groupby("Company").apply(lambda x: np.percentile(x, 75)))
    df_percentile_75["P75"] = df_percentile_75
    df_min = df_laptop[[column, "Company"]].groupby("Company").min()
    df_min["Min"] = df_min
    df_max = df_laptop[[column, "Company"]].groupby("Company").max()
    df_max["Max"] = df_max

    df_final = pd.concat(
        [df_min["Min"], df_percentile_25["P25"], df_mean["Mean"], df_percentile_75["P75"], df_max["Max"]], axis=1)
    df_final = df_final.sort_values("Mean", ascending=True, inplace=False)
    return df_final


# mean bar visualization
def df_mean_prices(values):
    df = df_laptop[values].value_counts()
    index = df.index
    list_mean = []
    for i in range(len(index)):
        value_mean = round(df_laptop[df_laptop[values] == index[i]]["Price_euros"].mean(), 2)
        list_mean.append(value_mean)

    list_index = list(index)
    df_final = pd.DataFrame(list(zip(list_index, list_mean)), columns=[values, "Price_mean"])
    df_final = df_final.sort_values("Price_mean")
    return df_final


# Palette
palette = sns.color_palette("Paired")
palette

fig = plt.figure(figsize=(20, 15))
ax = plt.axes()
plt.show()
sns.countplot(df_laptop["Company"], order=df_laptop["Company"].value_counts().index, palette=palette);
target = df_laptop["Company"].value_counts()
values = target.values
index = target.index
for i, g in enumerate(values):
    plt.text(i - 0.25, g, f"{g}", {"family": "serif", "size": 19, "weight": "semibold"})
fig.patch.set_facecolor("#A3A3A3")
ax.set_facecolor("#A3A3A3")
plt.text(-0.25, 350, "What are the companies used in this dataset and how many are there?",
         {"font": "family", "weight": "bold", "size": 25})
plt.text(-0.25, 340,
         "In this data frame,  Dell and Lenovo have the most pcs, both have 297 and the smallest are Huawei, Lg and Fujitsu, if we add the three the result is just 10.",
         {"font": "family", "size": 15})
plt.text(-0.25, 330, "This df has 17 companies spread in 1303 pcs.", {"font": "family", "size": 15})

ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
# simpson of company
Company = round(simpson(df_laptop["Company"]), 2)
Company


def typeNotebook(values):
    df = df_laptop[values].value_counts()
    values = df.values
    index = df.index
    lenght = len(df)
    squary_v = []
    for i in range(lenght):
        valuen = str(index[i]) + " \n" + "(" + str(values[i]) + ")"
        squary_v.append(valuen)

    return squary_v


labels = typeNotebook("TypeName")

fig = plt.figure(figsize=(10, 5))
ax = plt.axes()
ax.set_facecolor("#A3A3A3")
fig.patch.set_facecolor("#A3A3A3")
types = df_laptop["TypeName"].value_counts()
squarify.plot(sizes=types.values, label=labels, color=palette);
plt.text(-1, 120, "What are the types of pc used in this dataset and how many are there?",
         {"font": "family", "weight": "bold", "size": 15})
plt.text(-1, 114,
         "The type of PC that there is the most is Notebook with 727, and the second is Gaming with 205, both have almost 100 pcs ")
ax.spines[["left", "right", "top", "bottom"]].set_visible(False)

round(simpson(df_laptop["TypeName"]), 2)

labels = typeNotebook('Cpu_Mode')
fig = plt.figure(figsize=(17, 17))
ax = plt.axes()
fig.patch.set_facecolor("#A3A3A3")
ax.set_facecolor("#A3A3A3")
types = df_laptop["Cpu_Mode"].value_counts()
squarify.plot(sizes=types.values, label=labels, color=palette);
plt.text(0, 110, "What are the types of cpus in this dataset and how many are there?",
         {"font": "family", "weight": "bold", "size": 25});
plt.text(0, 107, "The presence is absolute of intel, from i7 to i3;the other closest processor is a9", {"size": 20});

Cpu_Mode = round(simpson(df_laptop["Cpu_Mode"]), 2)
Cpu_Mode

plt.show()

Ram = describe_df("Ram")
indice = Ram.index

fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
ax.hlines(y=indice, xmin=0, xmax=64, color='gray', alpha=0.5, linewidth=.5, linestyles='dashdot')
fig.patch.set_facecolor("#A3A3A3")
ax.set_facecolor("#A3A3A3")
plt.xticks(list(range(2, 70, 4)))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

for i, x in enumerate(Ram.Min):
    ax.scatter(x, i, color=palette[0])
for i, x in enumerate(Ram.P25):
    ax.scatter(x, i, color=palette[1])
for i, x in enumerate(Ram.Mean):
    ax.scatter(x, i, color=palette[2])
for i, x in enumerate(Ram.P75):
    ax.scatter(x, i, color=palette[3])
for i, x in enumerate(Ram.Max):
    ax.scatter(x, i, color=palette[5])

plt.text(-10, 25, "What is  the  most GB in Pcs?", {"font": "family", "weight": "bold", "size": 20});
plt.text(-10, 23.5, "The distribution is from the smallest mean gb  to the largest mean gb.", {"size": 12});
plt.text(0, 21.5, "| Min value", {"weight": "bold", "color": palette[0]});
plt.text(11, 21.5, "| Percentile 25", {"weight": "bold", "color": palette[1]});
plt.text(25, 21.5, "| Mean", {"weight": "bold", "color": palette[2]});
plt.text(32, 21.5, "| Percentile 75", {"weight": "bold", "color": palette[3]});
plt.text(46, 21.5, "| Max value", {"weight": "bold", "color": palette[5]});
descriptive("Ram")
plt.show()

Ghz = describe_df("GHZ")
indice = Ghz.index

fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
ax.hlines(y=indice, xmin=0, xmax=5, color='gray', alpha=0.5, linewidth=.5, linestyles='dashdot')
fig.patch.set_facecolor("#A3A3A3")
ax.set_facecolor("#A3A3A3")
plt.xticks(list(range(0, 5, 1)))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

for i, x in enumerate(Ghz.Min):
    ax.scatter(x, i, color=palette[0])
for i, x in enumerate(Ghz.P25):
    ax.scatter(x, i, color=palette[1])
for i, x in enumerate(Ghz.Mean):
    ax.scatter(x, i, color=palette[2])
for i, x in enumerate(Ghz.P75):
    ax.scatter(x, i, color=palette[3])
for i, x in enumerate(Ghz.Max):
    ax.scatter(x, i, color=palette[5])
plt.text(-1, 25, "What is the biggest GHZ in Pcs?", {"font": "family", "weight": "bold", "size": 20});
plt.text(-1, 23.5, "The distribution is from the smallest mean GHZ  to the largest mean GHZ.", {"size": 12});
plt.text(0, 21.5, "| Min value", {"weight": "bold", "color": palette[0]});
plt.text(0.85, 21.5, "| Percentile 25", {"weight": "bold", "color": palette[1]});
plt.text(1.9, 21.5, "| Mean", {"weight": "bold", "color": palette[2]});
plt.text(2.5, 21.5, "| Percentile 75", {"weight": "bold", "color": palette[3]});
plt.text(3.6, 21.5, "| Max value", {"weight": "bold", "color": palette[5]});
plt.show()

descriptive("GHZ")

Weight_kg = describe_df("Weight_kg")
indice = Weight_kg.index

fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
ax.hlines(y=indice, xmin=0, xmax=6, color='gray', alpha=0.5, linewidth=.5, linestyles='dashdot')
fig.patch.set_facecolor("#A3A3A3")
ax.set_facecolor("#A3A3A3")
plt.xticks(list(range(0, 6, 1)))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

for i, x in enumerate(Weight_kg.Min):
    ax.scatter(x, i, color=palette[0])
for i, x in enumerate(Weight_kg.P25):
    ax.scatter(x, i, color=palette[1])
for i, x in enumerate(Weight_kg.Mean):
    ax.scatter(x, i, color=palette[2])
for i, x in enumerate(Weight_kg.P75):
    ax.scatter(x, i, color=palette[3])
for i, x in enumerate(Weight_kg.Max):
    ax.scatter(x, i, color=palette[5])

plt.text(-1, 25, "What is the smallest Weight Kg in Pcs?", {"font": "family", "weight": "bold", "size": 20});
plt.text(-1, 23.5, "The distribution is from the smallest mean weight_kb  to the largest mean weight_kg.",
         {"size": 12});
plt.text(0, 21.5, "| Min value", {"weight": "bold", "color": palette[0]});
plt.text(1, 21.5, "| Percentile 25", {"weight": "bold", "color": palette[1]});
plt.text(2.3, 21.5, "| Mean", {"weight": "bold", "color": palette[2]});
plt.text(2.9, 21.5, "| Percentile 75", {"weight": "bold", "color": palette[3]});
plt.text(4.2, 21.5, "| Max value", {"weight": "bold", "color": palette[5]});

descriptive("Weight_kg")

fig = plt.figure(figsize=(20, 10))
ax = plt.axes()
fig.patch.set_facecolor("#A3A3A3")
ax.set_facecolor("#A3A3A3")
sns.countplot("MemoryType", hue="TypeName", data=df_laptop, palette=palette);
ax.spines[["top", "right"]].set_visible(False)
plt.legend([], [], frameon=False);
plt.text(-0.55, 450, "What is the  most common memory in the diferents types of PCs",
         {"font": "family", "weight": "bold", "size": 30});
plt.text(-0.55, 430, "The distribution is perfectly dominated by ssd memory and the second is HDD", {"size": 20});
plt.text(-0.55, 410, "| Ultrabook", {"weight": "bold", "color": palette[0], "size": 20});
plt.text(-0.09, 410, "| Notebook", {"weight": "bold", "color": palette[1], "size": 20});
plt.text(0.38, 410, "| Netbook", {"weight": "bold", "color": palette[2], "size": 20});
plt.text(0.8, 410, "| Gaming", {"weight": "bold", "color": palette[3], "size": 20});
plt.text(1.2, 410, "| 2 in 1 convertible", {"weight": "bold", "color": palette[4], "size": 20});
plt.text(2, 410, "| Workstation", {"weight": "bold", "color": palette[5], "size": 20});

Cpu_Mode = simpson(df_laptop["MemoryType"])
Cpu_Mode

# Create this new df because the extra memory its important.
df_extra_gb = df_laptop[df_laptop["ExtreGB"] > 0]

fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
fig.patch.set_facecolor("#A3A3A3")
ax.set_facecolor("#A3A3A3")
sns.countplot("ExtreGB", hue="Company", data=df_extra_gb, palette=palette);
ax.spines[["top", "right"]].set_visible(False)
plt.legend([], [], frameon=False);

plt.text(-0.8, 75, "What is the company with the most GB Extra", {"font": "family", "weight": "bold", "size": 18});
plt.text(-0.8, 70, "The distribution denotes the presence of 1000 GB extra is the most comun ");
plt.text(-0.8, 62.5, " | Lenovo", {"weight": "bold", "color": palette[0], "size": 13});
plt.text(-0.08, 62.5, " | Dell", {"weight": "bold", "color": palette[1], "size": 13});
plt.text(0.35, 62.5, " | Asus", {"weight": "bold", "color": palette[2], "size": 13});
plt.text(0.9, 62.5, " | MSI", {"weight": "bold", "color": palette[3], "size": 13});
plt.text(1.35, 62.5, " | HP", {"weight": "bold", "color": palette[4], "size": 13});
plt.text(1.72, 62.5, " | Samsung", {"weight": "bold", "color": palette[5], "size": 13});
plt.text(2.6, 62.5, " | Acer", {"weight": "bold", "color": palette[6], "size": 13});

fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
fig.patch.set_facecolor("#A3A3A3")
ax.set_facecolor("#A3A3A3")
sns.kdeplot(data=df_laptop, x="Price_euros", hue="MemoryType", multiple="fill", palette="pastel");
plt.text(-800, 1.3, "What is the density of each type of memory with  respect to the price euros",
         {"font": "family", "weight": "bold", "size": 12.5});
plt.text(-800, 1.2, "The SSD is the memory most expensive compared to other memories", {"font": "family", "size": 10});

fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
fig.patch.set_facecolor("#A3A3A3")
ax.set_facecolor("#A3A3A3")
sns.kdeplot(x="Price_euros", hue="HD", data=df_laptop, palette="pastel", fill=True);

ax.spines[["top", "right"]].set_visible(False)
plt.text(-1200, 0.00075, "If the pc is HD, how much influes  the price",
         {"font": "family", "weight": "bold", "size": 17.5});
plt.savefig('saved.jpeg')
plt.show()



sns.jointplot(x="Price_euros", y="Ram", data=df_laptop, hue="HD", palette="pastel");
plt.text(0,80, "a");
sns.jointplot(x="Price_euros", y="Inches", data=df_laptop, hue="Touchscreen", palette="pastel");

fig = plt.figure(figsize=(8,4))
ax = plt.axes()
fig.patch.set_facecolor("#A3A3A3")
ax.set_facecolor("#A3A3A3")

sns.scatterplot(x="Price_euros", y="Ram",hue="MemoryType",data=df_laptop);
ax.spines[["top", "right"]].set_visible(False)
plt.legend([],[], frameon=False);

plt.text(-900,85, " How the ram and type of memory  influes in the PCs price.", {"weight":"bold", "size":15});
plt.text(-900, 80, "All pcs with more 20GB of  Ram have SSD as  type of memory.");
plt.text(666, 72, "| SSD", {"color":"#2A7993", "weight":"bold", "size":15});
plt.text(1400, 72, "| Flash Storage", {"color":"#ED9723", "weight":"bold", "size":15});
plt.text(3300, 72, "| HDD", {"color":"#14AD2B", "weight":"bold", "size":15});
plt.text(4050, 72, "| Hybrid", {"color":"#EE2727", "weight":"bold", "size":15});

companys =df_mean_prices("Company")
fig = plt.figure(figsize=(8,4))
ax = plt.axes()
fig.patch.set_facecolor("#A3A3A3")
ax.set_facecolor("#A3A3A3")
plt.barh(companys.Company, companys.Price_mean, color= palette[0]);
ax.spines[["top", "right","bottom"]].set_visible(False)
plt.xticks([])
for i,g in enumerate(companys.Price_mean):
    plt.text(g,i-0.25, f"{g}", {"family":"serif", "size":10, "weight":"semibold"})

plt.text(-300,22, "What is the avarage price of PCs compared to companies?", {"weight":"bold","size":16});

cpu = df_mean_prices("Cpu")
fig = plt.figure(figsize=(8,4))
ax = plt.axes()
fig.patch.set_facecolor("#A3A3A3")
ax.set_facecolor("#A3A3A3")
plt.barh(cpu.Cpu, cpu.Price_mean, color= palette[2]);
ax.spines[["top", "right", "bottom"]].set_visible(False)
plt.xticks([])
for i, x in enumerate(cpu.Price_mean):
    plt.text(x+6.66, i, f"{x}", {"weight":"bold", "size":15})

plt.text(-100,3, "What is the avarage average price of PCs compared to cpu?", {"weight":"bold", "size":15});

opsys =df_mean_prices("OpSys")
fig = plt.figure(figsize=(10,5))
ax = plt.axes()
plt.barh(opsys.OpSys, opsys.Price_mean, data= df_laptop, color=palette[4]);
fig.patch.set_facecolor("#A3A3A3")
ax.set_facecolor("#A3A3A3")
ax.spines[["top", "right", "bottom"]].set_visible(False)
plt.xticks([]);
for y,x in enumerate(opsys.Price_mean):
    plt.text(x+6.66,y, f"{x}", {"weight":"bold", "size":12})

plt.text(-100,10, "What is the avarage average price of PCs compared to OpSys?", {"weight":"bold", "size":15});

Cpu_mode = df_mean_prices("Cpu_Mode")
fig = plt.figure(figsize=(12, 6))
ax = plt.axes()
plt.barh(Cpu_mode.Cpu_Mode, Cpu_mode.Price_mean, color=palette[6]);
fig.patch.set_facecolor("#A3A3A3")
ax.set_facecolor("#A3A3A3")
plt.xticks([])
ax.spines[["top", "right", "bottom"]].set_visible(False)

for y, x in enumerate(Cpu_mode.Price_mean):
    plt.text(x + 10, y - 0.2, f"{x}", {"weight": "bold", "size": 12})

plt.text(-250, 30, "What is the avarage  price of PCs compared to OpSys?", {"weight": "bold", "size": 20});


plt.show()
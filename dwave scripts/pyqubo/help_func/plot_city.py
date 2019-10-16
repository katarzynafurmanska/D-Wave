import networkx as nx
import matplotlib.pyplot as plt


def plot_city(cities, sol={}):

    cities_dict = dict(cities)
    G = nx.Graph()
    for city in cities_dict:
        G.add_node(city)

    # draw path
    if sol:
        city_order = []
        for i, v in sol.items():
            for j, v2 in v.items():
                if v2 == 1:
                    city_order.append(j)
        for i in range(len(city_order)):
            city1 = city_order[i]
            city2 = city_order[(i + 1) % len(city_order)]
            G.add_edge(city1, city2)

    plt.figure(figsize=(80,60))
    #pos = nx.spring_layout(G)
    nx.draw_networkx(G, cities_dict, node_size = 10000, font_size = 40, font_color = "white",width = 5)
    plt.axis("off")

    return plt

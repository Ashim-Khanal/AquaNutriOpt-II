This is a python package that find the optimal location(here in terms of Nodes/Reaches) for the placement of Best Management Practices and/or Technologies to reduce Nutrient loading in the Lake Okeechobee.
It takes two Files as input BMP_Tech.csv and Net_Data.csv. The specific format must be followed for this file which is explained in the Readme file of the foloowing github repository, https://github.com/Ashim-Khanal/AquaNutriOpt
It returns the total loading of the nutrients(Phosphorous, Nitrogens) at the end of the network (here the lake) and also provides the placement of possible BMPs or technologies at that reach within the given total budget.

AquNutriOpt v2.0 incorporates:
a. Time Indexed Model which considers multiple timeperiods nutrient loadings and transport to provide better and robust BMPs prescription.
b. Multi Objective Time Indexed Model where two measures Phosphorous and Nitrogen are to be minimized within provided budgets.


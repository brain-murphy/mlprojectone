Machine Learning project one: supervised learning

To compile the project to an executable jar, run:

gradle jar

This will produce a file called mlprojectone.jar in the /build/libs folder.
I've already compiled and placed the jar on the top level directory, so you can run the project
without compiling if you'd like.

Each figure and stat in my analysis is numbered. You can run the code that produced them by using the -t argument.
For example, to run the code that produced the data for figure 3, you would run

java -jar mlprojectone.jar -t 3

Wherever you run this jar from, it is important that you have /Iris.csv and /propaneData.ser
in the same directory.
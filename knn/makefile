############################# Makefile ##########################
all: util.o distances.o fileUtils.o knn.o
util.o: util.cpp
	g++ -c util.cpp -o bin/util.o
distances.o: distances.cpp
	g++ -c distances.cpp -o bin/distances.o
fileUtils.o: fileUtils.cpp
	g++ -c fileUtils.cpp -o bin/fileUtils.o
knn.o: knn.cpp
	g++ -c knn.cpp -o bin/knn.o
	g++ bin/distances.o bin/util.o bin/fileUtils.o bin/knn.o -o bin/knn
	sudo chmod +x bin/knn
clean:
	rm -rf bin/*
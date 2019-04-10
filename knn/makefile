############################# Makefile ##########################
all: util.o distances.o fileUtils.o knn.o main.o
util.o: util.cpp
	g++ -c util.cpp -o bin/util.o
distances.o: distances.cpp
	g++ -c distances.cpp -o bin/distances.o
fileUtils.o: fileUtils.cpp
	g++ -c fileUtils.cpp -o bin/fileUtils.o
knn.o: knn.cpp
	g++ -c knn.cpp -o bin/knn.o	
main.o: main.cpp
	g++ -c main.cpp -o bin/main.o
	g++ bin/distances.o bin/util.o bin/fileUtils.o bin/knn.o bin/main.o -o bin/main
	sudo chmod +x bin/main
clean:
	rm -rf bin/*
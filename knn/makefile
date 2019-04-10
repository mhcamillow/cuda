############################# Makefile ##########################
all: util.o knn.o
util.o: util.cpp
	g++ -c util.cpp -o bin/util.o
knn.o: knn.cpp
	g++ -c knn.cpp -o bin/knn.o
	g++ bin/util.o bin/knn.o -o bin/knn
	sudo chmod +x bin/knn
clean:
	rm -rf bin/*
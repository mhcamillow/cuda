class FileUtils
{
    private:
        static void parseFeatures(std::string line, float * features);
    public:
        static int getNumberOfElements(char *filepath);
        static int getNumberOfFeatures(char *filepath);
        static void loadFeatures(char *filepath, float * features, int feature_count);
        static void loadFile(char *filepath, int count, int * labels, float * features, int feature_count);
        static void loadLabels(char *filepath, int * labels);
};
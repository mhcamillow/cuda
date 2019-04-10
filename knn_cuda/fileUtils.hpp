class FileUtils
{
    public:
        static int getNumberOfElements(char *filepath);
        static int getNumberOfFeatures(char *filepath);
        static void loadFeatures(std::string line, double * features);
        static void loadFile(char *filepath, int count, int * labels, double * features, int feature_count);
};
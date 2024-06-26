#include "input_output.h"


// Function to print harmonic drive values
void print_harmonic_drive_values(HarmonicDriveParameterMap &harmonic_drive_values)
{
    Logger::info("Harmonic Drive Values: (units are m/coil and m)");
    for (const auto &value : harmonic_drive_values)
    {
        Logger::info(value.first + ": " + to_string(value.second));
    }
}

// Function to get a single character input without echoing to the console (POSIX)
char getch()
{
    char buf = 0;
    struct termios old = {0};
    if (tcgetattr(STDIN_FILENO, &old) < 0)
        perror("tcsetattr()");
    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 1;
    old.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &old) < 0)
        perror("tcsetattr ICANON");
    if (read(STDIN_FILENO, &buf, 1) < 0)
        perror("read()");
    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;
    if (tcsetattr(STDIN_FILENO, TCSADRAIN, &old) < 0)
        perror("tcsetattr ~ICANON");
    return buf;
}

// Function to ask the user if they want to proceed
bool askUserToProceed()
{
    std::string input;
    std::cout << "The harmonic drive values above will be optimized to achieve bn values within the maximum value specified above. Do you want to proceed with the optimization? (Y/n): ";
    std::getline(std::cin, input);
    return input == "Y" || input == "y";
}

// Function to get user input with a default value
double getUserInput(const std::string &prompt, double default_value)
{
    double value = 0;
    std::string input;

    while (value == 0)
    {
        std::cout << prompt << " (default: " << default_value << "): ";
        std::getline(std::cin, input);

        if (input.empty())
        {
            value = default_value;
        }
        else
        {
            try
            {
                value = std::stod(input);
                if (value <= 0)
                {
                    value = 0;
                    std::cerr << "Input must be greater than 0. Please try again." << std::endl;
                }
            }
            catch (...)
            {
                std::cerr << "Invalid input. Please enter a valid number." << std::endl;
            }
        }
    }

    Logger::info("Using " + std::to_string(value) + " as maximum absolute bn value.");

    return value;
}

// Function to display files in a directory and allow user to select one
boost::filesystem::path selectJsonFile()
{
    boost::filesystem::path dir_path(DATA_DIR_PATH);
    std::vector<boost::filesystem::path> json_files;

    if (boost::filesystem::exists(dir_path) && boost::filesystem::is_directory(dir_path))
    {
        for (const auto &entry : boost::filesystem::directory_iterator(dir_path))
        {
            if (boost::filesystem::is_regular_file(entry) && entry.path().extension() == ".json")
            {
                json_files.push_back(entry.path());
            }
        }
    }

    if (json_files.empty())
    {
        throw std::runtime_error("No JSON files found in the " + DATA_DIR_PATH + " directory. Please add the JSON file of the model you wish to optimize there.");
    }

    // convert file names to a string
    std::vector<std::string> json_file_names;
    for (const auto& file : json_files)
    {
        json_file_names.push_back(file.filename().string());
    }

    // let the user select a file
    std::string prompt = "Select the JSON file for the model you wish to optimize. If your model is not in the list, make sure it is placed in the " + DATA_DIR_PATH + " directory.";
    int selected_index = selectFromList(json_file_names, prompt);

    Logger::info("Selected model: " + json_files[selected_index].string());

    return json_files[selected_index];
}

// displayes strings in the terminal and lets the user select one. Returns the selected index
int selectFromList(std::vector<std::string> options, std::string user_prompt){
    int selected_index = 0;
    char key;
    while (true)
    {
        system("clear"); // Clear the terminal screen on POSIX systems
        std::cout << user_prompt << std::endl;
        std::cout << "Use arrow keys and enter to select."<< std::endl;
        for (size_t i = 0; i < options.size(); ++i)
        {
            if (i == selected_index)
            {
                std::cout << "> " << options[i] << std::endl;
            }
            else
            {
                std::cout << "  " << options[i] << std::endl;
            }
        }

        key = getch();

        if (key == '\033')
        { // arrow keys for POSIX
            getch();    // skip the [
            switch (getch())
            {
            case 'A': // up
                if (selected_index > 0)
                    selected_index--;
                break;
            case 'B': // down
                if (selected_index < options.size() - 1)
                    selected_index++;
                break;
            }
        }
        else if (key == '\r' || key == '\n')
        {
            break;
        }
    }

    return selected_index;
}

// Function to copy model from src to the build dir with appending timestamp
void copyModelWithTimestamp(const boost::filesystem::path &src_path)
{
    if (!boost::filesystem::exists(src_path))
    {
        Logger::error("Source file does not exist: " + src_path.string());
        return;
    }

    std::string filename = src_path.filename().string();
    auto now = std::chrono::system_clock::now();
    auto now_sec = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    std::string base_filename = src_path.stem().string();  // filename without extension
    std::string extension = src_path.extension().string(); // file extension

    std::string new_filename = base_filename + "_" + std::to_string(now_sec) + extension;
    boost::filesystem::path dest_path = MODEL_OUTPUT_DIR + new_filename;

    try
    {
        boost::filesystem::create_directory(MODEL_OUTPUT_DIR);
        boost::filesystem::copy_file(src_path, dest_path);

        // add the build path for clarity
        std::string modified_dest_path = dest_path.string();
        modified_dest_path.insert(1, "/build");

        Logger::info("The optimized model has been exported to: " + modified_dest_path);
    }
    catch (const boost::filesystem::filesystem_error &e)
    {
        Logger::error("Error while exporting optimized model: " + std::string(e.what()));
        Logger::error("The optimized model has instead been saved to: " + src_path.string());
    }
}

// exports a vector to a csv with ascending indexing
void export_vector_to_csv(const std::vector<double>& vector, const std::string& csv_path)
{
    std::vector<std::pair<double, double>> indexed_vector;
    for (size_t i = 0; i < vector.size(); ++i)
    {
        indexed_vector.push_back(std::make_pair(static_cast<double>(i), vector[i]));
    }

    export_data_to_csv(indexed_vector, csv_path);
}

// exports a vector of pairs to a csv
void export_data_to_csv(const std::vector<std::pair<double, double>>& vector, const std::string& csv_path)
{
    std::ofstream csv_file(csv_path);
    if (!csv_file)
    {
        Logger::error("Failed to open CSV file: " + csv_path);
        return;
    }

    // Write the header
    csv_file << "Index,Value\n";

    for (size_t i = 0; i < vector.size(); ++i)
    {
        csv_file << vector[i].first << "," << vector[i].second << "\n";
    }
    csv_file.close();

    Logger::info("Vector exported to CSV file: " + csv_path);
}

// Function to print values from a vector
void print_vector(const std::vector<double>& data, std::string label)
{
    Logger::info(label + " values:");
    for (size_t i = 0; i < data.size(); ++i)
    {
        Logger::info(label + "[" + std::to_string(i + 1) + "] = " + std::to_string(data[i]));
    }
}
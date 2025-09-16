#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
/* 

------ This is the model Python code being replaced ------

if dcf_implied_price < price_tmr:

    delta = price_tmr - dcf_implied_price - 90

    print('\n')
    print(f'Stock is overvalued by ${delta}. Recomended action: SHORT.')
    print('\n')



elif dcf_implied_price > price_tmr:

    delta = dcf_implied_price - price_tmr

    print('\n')
    print(f'Stock is undervalued by ${delta}. Recomended action: LONG.')
    print('\n')

    

else:
    print('\n')
    print(f'Stock is correctly priced. No recomended trading action.')
    print('\n')

*/


// This is a standard main function that accepts command-line arguments
int main(int argc, char* argv[]) {
    // We expect 3 arguments: 1. program name, 2. dcf_price, 3. tmr_price
    if (argc != 3) {
        // Print errors to std::cerr
        std::cerr << "Error: Invalid number of arguments." << std::endl;
        return 1; // Exit with an error code
    }

    // Convert the text arguments from the command line into numbers (doubles)
    double dcf_implied_price = std::stod(argv[1]);
    double price_tmr = std::stod(argv[2]);
    double delta = 0.0;

    // The core logic
    if (dcf_implied_price < price_tmr) {
        delta = price_tmr - dcf_implied_price - 90;
        // Print the final result to the console (standard output)
        std::cout << "Stock is overvalued by $" << std::fixed << std::setprecision(2) << delta << ". Recommended action: SHORT.";
    } else if (dcf_implied_price > price_tmr) {
        delta = dcf_implied_price - price_tmr;
        std::cout << "Stock is undervalued by $" << std::fixed << std::setprecision(2) << delta << ". Recommended action: LONG.";
    } else {
        std::cout << "Stock is correctly priced. No recommended trading action.";
    }

    return 0; // Indicate successful execution
}
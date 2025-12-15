////////////////////////////////////////////////////////////////////////////////////////////////
//                    Use Eigen library as the linear equation solver                         //
//                add static plummer baryons to the NFW dark matter halo                      //
////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <string>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <memory>
#include <unordered_map>
#include <optional>
#include <algorithm>
#include <cctype>
#include <Eigen/Dense>

// Forward declarations
class SimulationParameters;
class SimulationState;
class FileManager;
class Logger;

// Simulation configuration
// total step to run the simulation
constexpr int DEFAULT_TOTAL_STEPS = 1000000000;
// save step to save the simulation state
constexpr int DEFAULT_SAVE_STEPS = 1000;
// default iteration steps for relaxation
constexpr int DEFAULT_RELAXATION_STEPS = 10;
// default density threshold for stopping the simulation
constexpr double DEFAULT_DENSITY_THRESHOLD = 1e30;


/**
 * Logger class to handle output messages
 */
class Logger {
private:
    bool verbose;
    
public:
    explicit Logger(bool isVerbose = false) : verbose(isVerbose) {}
    
    void setVerbose(bool isVerbose) { verbose = isVerbose; }
    
    void info(const std::string& message) const {
        std::cout << message << std::endl;
    }
    void debug(const std::string& message) const {
        if (verbose) std::cout << "[DEBUG] " << message << std::endl;
    }
    void error(const std::string& message) const {
        std::cerr << "[ERROR] " << message << std::endl;
    }
};

/**
 * Class to handle simulation parameters
 * NOTE: 物理参数不再给默认值，必须由 Basic 文件提供
 */
class SimulationParameters {
public:
    // Simulation control parameters (这些保留默认，不影响“必须由Basic驱动”的要求)
    int totalStep;
    int saveStep;
    double totalTime;

    // 必须从 Basic 赋值:
    double a;
    double c;
    double sigma_0;
    double omega;
    double brem_prefactor;
    double mass_norm;
    double scale_norm;
    double age_of_universe;
    double epsilon;
    bool if_brem;
    bool if_anni;

    // IO parameters（必须由 Basic 决定）
    std::string tag;
    std::string inputDir;
    std::string outputFile;

    SimulationParameters()
        : totalStep(DEFAULT_TOTAL_STEPS),
          saveStep(DEFAULT_SAVE_STEPS),
          // epsilon(DEFAULT_EPSILON),
          totalTime(0.0)
    {
        epsilon = 0.0;
        if_brem = false;
        if_anni = false;
        // 重要：不再设置 a,c,sigma_0,mass_norm,scale_norm,tag,inputDir,outputFile
        // 这些都必须在 load_from_basic() 里读取并赋值
        // 初始化 if_brem 和 if_anni
    }
    
    void display(const Logger& logger) const {
        logger.info("Initial profile: " + tag);
        logger.info("Initial time: " + std::to_string(totalTime));
        logger.info("Total step: " + std::to_string(totalStep));
        logger.info("Save step: " + std::to_string(saveStep));
        logger.info("Abs(delta u/u): " + std::to_string(epsilon));
        logger.info("Cross section (sigma_0): " + std::to_string(sigma_0));
        logger.info("w = m_phi / m_chi: " + std::to_string(omega));
        logger.info("Conduction parameter (a, c): " + std::to_string(a) + ", " + std::to_string(c));
        logger.info("Baryon parameter (mass_norm, scale_norm): " + std::to_string(mass_norm) + ", "
                 + std::to_string(scale_norm));
        logger.info(std::string("Flags (if_brem, if_anni): ") + (if_brem ? "1" : "0") + ", " + (if_anni ? "1" : "0"));
    }
    bool checkForAgeOfUniverse(Logger& logger) const {
        if (totalTime > age_of_universe) {
            logger.info("It has been longer than the age of universe!");
            return true;
        }
        return false;
    }
};

// Baryon enclosed mass function implementations
// Plummer model, the default one
double MbaryonP(double myr, double mymass, double myscale) {
    return mymass * std::pow(1.0 + (myscale * myscale)/(myr * myr), -1.5);
}
// Hernquist model
double MbaryonH(double myr, double mymass, double myscale) {
    return mymass * std::pow(1.0 + myscale/myr, -2.0);
}
// Single-Power-Law model
double MbaryonSPL(double myr, double mymass) {
    return mymass * std::pow(myr, 0.6);
}

/**
 * Class to handle file operations
 */
class FileManager {
private:
    Logger& logger;

public:
    explicit FileManager(Logger& log) : logger(log) {}
    
    // Read matrix from file with error handling
    Eigen::MatrixXd readMatrix(const std::string& filename) const {
        std::vector<double> buffer;
        int cols = 0, rows = 0;
        
        std::ifstream infile(filename);
        if (!infile) throw std::runtime_error("Cannot open file: " + filename);
        
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream stream(line);
            int temp_cols = 0;
            double value;
            while (stream >> value) {
                buffer.push_back(value);
                temp_cols++;
            }
            if (temp_cols == 0) continue;
            if (cols == 0) cols = temp_cols;
            rows++;
        }
        infile.close();
        
        Eigen::MatrixXd result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i, j) = buffer[cols * i + j];
            }
        }
        return result;
    }
    
    // Write simulation parameters to file
    void writeParameters(const SimulationParameters& params, 
                         const class SimulationState& state) const;
    
    // Write simulation state to file
    void writeState(const std::string& filename, 
                    double totalTime, 
                    int step, 
                    const class SimulationState& state) const;
};

/**
 * Class to manage the simulation state
 */
class SimulationState {
public:
    Eigen::ArrayXd RList;      // Radius
    Eigen::ArrayXd RhoList;    // Density
    Eigen::ArrayXd MList;      // Mass (dark matter only)
    Eigen::ArrayXd MhyList;    // Total mass (dark matter + baryon)
    Eigen::ArrayXd uList;      // Specific internal energy
    Eigen::ArrayXd LList;      // Luminosity
    Eigen::ArrayXd CList;      // Cooling rate
    Eigen::ArrayXd vList;      // 1D Velocity dispersion
    Eigen::ArrayXd pList;      // Pressure
    Eigen::ArrayXd aList;      // Adiabatic variable
    int NoLayers;              // Number of layers
    
    SimulationState() : NoLayers(0) {}
    
    void initialize(const SimulationParameters& params, 
                    const FileManager& fileManager, 
                    Logger& logger) {
        try {
            // Construct input file paths by Basic-dir + tag
            std::string nameR   = params.inputDir + "RList-"   + params.tag + ".txt";
            std::string nameRho = params.inputDir + "RhoList-" + params.tag + ".txt";
            std::string nameM   = params.inputDir + "MList-"   + params.tag + ".txt";
            std::string nameu   = params.inputDir + "uList-"   + params.tag + ".txt";
            std::string nameL   = params.inputDir + "LList-"   + params.tag + ".txt";
            std::string nameC   = params.inputDir + "CList-"   + params.tag + ".txt";
            
            RList   = fileManager.readMatrix(nameR).array();
            RhoList = fileManager.readMatrix(nameRho).array();
            MList   = fileManager.readMatrix(nameM).array();
            uList   = fileManager.readMatrix(nameu).array();
            LList   = fileManager.readMatrix(nameL).array();
            CList   = fileManager.readMatrix(nameC).array();
            
            NoLayers = RList.rows();
            
            MhyList = Eigen::ArrayXd::Zero(NoLayers);
            vList   = Eigen::ArrayXd::Zero(NoLayers);
            pList   = Eigen::ArrayXd::Zero(NoLayers);
            aList   = Eigen::ArrayXd::Zero(NoLayers);
            
            vList = std::sqrt(2.0/3.0) * uList.sqrt();
            
            // Add baryon mass (Plummer)
            for (int i = 0; i < (NoLayers-1); i++) {
                MhyList(i) = MList(i) + MbaryonP(RList(i), params.mass_norm, params.scale_norm);
            }
            
            logger.info("Number of layers: " + std::to_string(NoLayers));
            logger.info("Initial r inner most: " + std::to_string(RList(0)));
            logger.info("Initial r next to outer most: " + std::to_string(RList(NoLayers-1)));
            logger.info("Initial rho inner most: " + std::to_string(RhoList(0)));
            logger.info("Initial rho next to outer most: " + std::to_string(RhoList(NoLayers-1)));
        } catch (const std::exception& e) {
            logger.error("Failed to initialize simulation state: " + std::string(e.what()));
            throw;
        }
    }
    
    bool checkForAbnormalState(Logger& logger) const {
        if (RhoList(0) > DEFAULT_DENSITY_THRESHOLD) {
            logger.info("Rho reaches threshold!");
            return true;
        }
        if (RList(0) < 0) {
            logger.info("R(0) is negative!");
            return true;
        }
        if (std::isnan(RList(0))) {
            logger.info("R is nan!");
            return true;
        }
        return false;
    }
};


/**
 * Class handling the simulation evolution
 */
class Simulator {
private:
    SimulationParameters params;
    SimulationState state;
    FileManager fileManager;
    Logger logger;
    
    // Helper arrays for evolution calculations
    Eigen::ArrayXd deltaUcoeff;
    Eigen::ArrayXd deltaU;
    Eigen::ArrayXd hydrostaticI;
    Eigen::ArrayXd hydrostaticM;
    Eigen::ArrayXd hydrostaticF;
    Eigen::MatrixXd Hydromat;
    Eigen::VectorXd Hydrob;
    Eigen::VectorXd deltaR;
    Eigen::VectorXd deltap;
    Eigen::VectorXd deltaRho;
    // Lookup tables for big_int(vd) (dimensionless in units of v_fid)
    Eigen::ArrayXd vd_grid;  // vd grid
    Eigen::ArrayXd bi_grid;  // big_int(vd) on that grid
    Eigen::ArrayXd bm_grid;  // brem_int(vd) on that grid

public:
    Simulator(const SimulationParameters& p, 
              const FileManager& fm, 
              Logger& log) : 
        params(p), 
        fileManager(fm), 
        logger(log) {}
    
    void initialize() {
        state.initialize(params, fileManager, logger);

        // Load precomputed big_int(vd) lookup tables (vdiList-/biList-)
        loadBigIntTable();
        
        int NoLayers = state.NoLayers;
        deltaUcoeff = Eigen::ArrayXd::Zero(NoLayers);
        deltaU      = Eigen::ArrayXd::Zero(NoLayers);
        hydrostaticI= Eigen::ArrayXd::Zero(NoLayers);
        hydrostaticM= Eigen::ArrayXd::Zero(NoLayers);
        hydrostaticF= Eigen::ArrayXd::Zero(NoLayers);
        
        Hydromat = Eigen::MatrixXd::Zero((NoLayers-1), (NoLayers-1));
        Hydrob   = Eigen::VectorXd::Zero(NoLayers-1);
        deltaR   = Eigen::VectorXd::Zero(NoLayers-1);
        deltap   = Eigen::VectorXd::Zero(NoLayers-1);
        deltaRho = Eigen::VectorXd::Zero(NoLayers-1);
        
        std::ofstream file(params.outputFile, std::ofstream::out | std::ofstream::app);
        if (file.is_open()) {
            file << "Input profile name: " << params.tag << '\n'
                 << "Initial time: "  << params.totalTime << '\n'
                 << "Number of layers: " << state.NoLayers << '\n'
                 << "Initial r inner most: " << state.RList(0) << '\n'
                 << "Initial r outer most: " << state.RList(NoLayers-1) << '\n'
                 << "Initial rho inner most: " << state.RhoList(0) << '\n'
                 << "Initial rho outer most: " << state.RhoList(NoLayers-1) << '\n'
                 << "Total step: " << params.totalStep << '\n'
                 << "Save step: " << params.saveStep << '\n'
                 << "Abs(delta u/u): " << params.epsilon << '\n'
                 << "Cross section (sigma_0): " << params.sigma_0 << '\n'
                 << "Conduction parameter (a, c): " << params.a << " , " << params.c << '\n'
                 << "Baryon parameter (massnorm, scalenorm): " << params.mass_norm << ", " << params.scale_norm << '\n'
                 << "time, step, SIDM radius, SIDM density, SIDM enclosed mass, SIDM internal energy, SIDM luminosity" << '\n'
                 << std::scientific << std::setprecision(10) << params.totalTime << " " << 0 << '\n'
                 << state.RList.transpose() << '\n'
                 << state.RhoList.transpose() << '\n'
                 << state.MList.transpose() << '\n'
                 << state.uList.transpose() << '\n'
                 << state.LList.transpose() << '\n'
                 << state.CList.transpose() << '\n';
        }
        file.close();
        
        logger.info("Evolution with baryon starts!");
    }
    
    void runSimulation() {
        std::ofstream outputFile(params.outputFile, std::ofstream::out | std::ofstream::app);
        if (!outputFile.is_open()) {
            throw std::runtime_error("Failed to open output file: " + params.outputFile);
        }
        
        int currentSaveStep = params.saveStep;
        
        for (int tstep = 1; tstep < (params.totalStep + 1); tstep++) {
            try {
                double deltat = performConductionStep();
                params.totalTime += deltat;
                performRelaxationStep();
                updateProfiles();
                
                if (state.RhoList(0) >= 1e6 && currentSaveStep != 1) {
                    currentSaveStep = 1;
                    logger.info("Density threshold 1e6 reached! Save frequency increased to every step.");
                }
                if (state.checkForAbnormalState(logger)) break;  // 检查是否中心密度已爆炸

                if (params.checkForAgeOfUniverse(logger))
                break;  // 检查是否已到宇宙年龄
                
                if ((tstep % currentSaveStep) == 0) {
                    saveResults(outputFile, tstep);
                    logger.debug("Saved at time = " + std::to_string(params.totalTime) + 
                                 ", step = " + std::to_string(tstep));
                }
            } catch (const std::exception& e) {
                logger.error("Error during simulation step " + std::to_string(tstep) + 
                             ": " + std::string(e.what()));
                break;
            }
        }
        
        outputFile.close();
        logger.info("Evolution ends");
    }
    
private:
    double performConductionStep() {
        int NoLayers = state.NoLayers;
        
        deltaUcoeff(0) = - ((state.LList(0) / state.MList(0))) / state.uList(0) + state.CList(0) / state.RhoList(0);
        for (int i = 1; i < (NoLayers-1); i++) {
            deltaUcoeff(i) = -((state.LList(i) - state.LList(i-1)) / (state.MList(i) - state.MList(i-1)) 
                               ) / state.uList(i) + state.CList(i) / state.RhoList(i);
        }
        
        double deltat = params.epsilon / (deltaUcoeff.abs().maxCoeff());
        
        deltaU = deltaUcoeff * state.uList * deltat;
        state.uList += deltaU;
        
        state.pList = (2.0/3.0) * (state.RhoList * state.uList);
        state.aList = (2.0/3.0) * (state.RhoList.pow(-2.0/3.0) * state.uList);
        
        return deltat;
    }
    
    void performRelaxationStep() {
        int NoLayers = state.NoLayers;
        
        for (int hstep = 0; hstep < DEFAULT_RELAXATION_STEPS; hstep++) {
            try {
                setupHydrostaticMatrix();
                deltaR = Hydromat.llt().solve(Hydrob);
                calculateDeltaRhoAndP();
                for (int i = 0; i < (NoLayers-1); i++) {
                    state.RhoList(i) += deltaRho(i);
                    state.pList(i)   += deltap(i);
                    state.RList(i)   += deltaR(i);
                }
            } catch (const std::exception& e) {
                logger.error("Error during relaxation step: " + std::string(e.what()));
                throw;
            }
        }
    }
    
    void setupHydrostaticMatrix() {
        int NoLayers = state.NoLayers;
        
        double R0 = state.RList(0);
        double R0_2 = R0 * R0;
        double R0_3 = R0_2 * R0;
        double R0_4 = R0_2 * R0_2;
        double R1 = state.RList(1);
        double R1_2 = R1 * R1;
        double R1_3 = R1_2 * R1;
        double R1_R0 = R1 - R0;
        double R1_R0_sum_of_squares = R1_2 + R1*R0 + R0_2;
        double R1_3_minus_R0_3_inv = 1.0 / (R1_R0 * R1_R0_sum_of_squares);
        
        Hydromat(0, 0) = 8.0 * R0 * (state.pList(1) - state.pList(0)) + 
                        20.0 * R0_4 * 
                        (state.pList(0) / R0_3 + 
                        state.pList(1) * R1_3_minus_R0_3_inv) + 
                        3.0 * state.MhyList(0) * R1 * R0_2 * 
                        (-state.RhoList(0) / R0_3 + 
                        state.RhoList(1) * R1_3_minus_R0_3_inv);
        
        Hydromat(0, 1) = state.MhyList(0) * (state.RhoList(0) + state.RhoList(1)) - 
                        20.0 * R0_2 * state.pList(1) * 
                        R1_2 * R1_3_minus_R0_3_inv - 
                        3.0 * state.MhyList(0) * state.RhoList(1) * R1_3 * 
                        R1_3_minus_R0_3_inv;
        
        Hydrob(0) = -4.0 * R0_2 * (state.pList(1) - state.pList(0)) - 
                    state.MhyList(0) * (state.RhoList(0) + state.RhoList(1)) * R1;
        
        for (int i = 1; i < (NoLayers-2); ++i) {
            double Ri = state.RList(i);
            double Ri_2 = Ri * Ri;
            double Ri_3 = Ri_2 * Ri;
            double Ri_4 = Ri_2 * Ri_2;
            double Ri_1 = state.RList(i-1);
            double Ri_1_2 = Ri_1 * Ri_1;
            double Ri_1_3 = Ri_1_2 * Ri_1;
            double Ri1 = state.RList(i+1);
            double Ri1_2 = Ri1 * Ri1;
            double Ri1_3 = Ri1_2 * Ri1;
            double Ri_Ri_1 = Ri - Ri_1;
            double Ri_Ri_1_sum_of_squares = Ri_2 + Ri*Ri_1 + Ri_1_2;
            double Ri_3_minus_Ri_1_3_inv = 1.0 / (Ri_Ri_1 * Ri_Ri_1_sum_of_squares);
            double Ri1_Ri = Ri1 - Ri;
            double Ri1_Ri_sum_of_squares = Ri1_2 + Ri1*Ri + Ri_2;
            double Ri1_3_minus_Ri_3_inv = 1.0 / (Ri1_Ri * Ri1_Ri_sum_of_squares);
            
            Hydromat(i, i-1) = -state.MhyList(i) * (state.RhoList(i) + state.RhoList(i+1)) - 
                              20.0 * Ri_2 * state.pList(i) * 
                              Ri_1_2 * Ri_3_minus_Ri_1_3_inv + 
                              3.0 * state.MhyList(i) * (Ri1 - Ri_1) * 
                              state.RhoList(i) * Ri_1_2 * 
                              Ri_3_minus_Ri_1_3_inv;
            
            Hydromat(i, i) = 8.0 * Ri * (state.pList(i+1) - state.pList(i)) + 
                            20.0 * Ri_4 * 
                            (state.pList(i) * Ri_3_minus_Ri_1_3_inv + 
                            state.pList(i+1) * Ri1_3_minus_Ri_3_inv) + 
                            3.0 * state.MhyList(i) * (Ri1 - Ri_1) * 
                            Ri_2 * 
                            (-state.RhoList(i) * Ri_3_minus_Ri_1_3_inv + 
                            state.RhoList(i+1) * Ri1_3_minus_Ri_3_inv);
            
            Hydromat(i, i+1) = state.MhyList(i) * (state.RhoList(i) + state.RhoList(i+1)) - 
                              20.0 * Ri_2 * state.pList(i+1) * 
                              Ri1_2 * Ri1_3_minus_Ri_3_inv - 
                              3.0 * state.MhyList(i) * (Ri1 - Ri_1) * 
                              state.RhoList(i+1) * Ri1_2 * 
                              Ri1_3_minus_Ri_3_inv;
            
            Hydrob(i) = -4.0 * Ri_2 * (state.pList(i+1) - state.pList(i)) - 
                        state.MhyList(i) * (state.RhoList(i) + state.RhoList(i+1)) * 
                        (Ri1 - Ri_1);
        }
        
        int last = NoLayers - 2;
        double Rl = state.RList(last);
        double Rl_2 = Rl * Rl;
        double Rl_3 = Rl_2 * Rl;
        double Rl_4 = Rl_2 * Rl_2;
        double Rl_1 = state.RList(last-1);
        double Rl_1_2 = Rl_1 * Rl_1;
        double Rl_1_3 = Rl_1_2 * Rl_1;
        double Rl1 = state.RList(last+1);
        double Rl_Rl_1 = Rl - Rl_1;
        double Rl_Rl_1_sum_of_squares = Rl_2 + Rl*Rl_1 + Rl_1_2;
        double Rl_3_minus_Rl_1_3_inv = 1.0 / (Rl_Rl_1 * Rl_Rl_1_sum_of_squares);
        
        Hydromat(last, last-1) = -state.MhyList(last) * 
                               (state.RhoList(last) + state.RhoList(last+1)) - 
                               20.0 * Rl_2 * 
                               state.pList(last) * Rl_1_2 * 
                               Rl_3_minus_Rl_1_3_inv + 
                               3.0 * state.MhyList(last) * 
                               (Rl1 - Rl_1) * 
                               state.RhoList(last) * Rl_1_2 * 
                               Rl_3_minus_Rl_1_3_inv;
        
        Hydromat(last, last) = 8.0 * Rl * 
                             (state.pList(last+1) - state.pList(last)) + 
                             20.0 * Rl_4 * 
                             state.pList(last) * Rl_3_minus_Rl_1_3_inv - 
                             3.0 * state.MhyList(last) * 
                             (Rl1 - Rl_1) * 
                             Rl_2 * state.RhoList(last) * 
                             Rl_3_minus_Rl_1_3_inv;
        
        Hydrob(last) = -4.0 * Rl_2 * 
                      (state.pList(last+1) - state.pList(last)) - 
                      state.MhyList(last) * (state.RhoList(last) + 
                      state.RhoList(last+1)) * 
                      (Rl1 - Rl_1);
    }
    
    void calculateDeltaRhoAndP() {
        int NoLayers = state.NoLayers;
        
        double R0 = state.RList(0);
        double R0_2 = R0 * R0;
        double R0_3 = R0_2 * R0;
        
        deltaRho(0) = -3.0 * state.RhoList(0) * R0_2 * deltaR(0) / R0_3;
        deltap(0)   = -5.0 * state.pList(0) * R0_2 * deltaR(0) / R0_3;
        
        for (int i = 1; i < (NoLayers-1); i++) {
            double Ri = state.RList(i);
            double Ri_2 = Ri * Ri;
            double Ri_3 = Ri_2 * Ri;
            double Ri_1 = state.RList(i-1);
            double Ri_1_2 = Ri_1 * Ri_1;
            double Ri_1_3 = Ri_1_2 * Ri_1;
            double Ri_Ri_1 = Ri - Ri_1;
            double Ri_Ri_1_sum_of_squares = Ri_2 + Ri*Ri_1 + Ri_1_2;
            double Ri_3_minus_Ri_1_3_inv = 1.0 / (Ri_Ri_1 * Ri_Ri_1_sum_of_squares);
            
            deltaRho(i) = -3.0 * state.RhoList(i) *
                          (Ri_2 * deltaR(i) - Ri_1_2 * deltaR(i-1)) * 
                          Ri_3_minus_Ri_1_3_inv;
            deltap(i)   = -5.0 * state.pList(i) *
                          (Ri_2 * deltaR(i) - Ri_1_2 * deltaR(i-1)) * 
                          Ri_3_minus_Ri_1_3_inv;
        }
    }
    
    void updateProfiles() {
        int NoLayers = state.NoLayers;
        
        for (int i = 0; i < (NoLayers-1); i++) {
            state.MhyList(i) = state.MList(i) + MbaryonP(state.RList(i), params.mass_norm, params.scale_norm);
        }
        
        state.uList = (1.5) * state.aList * state.RhoList.pow(2.0/3.0);
        state.vList = std::sqrt(2.0/3.0) * state.uList.sqrt();
        
        double R0 = state.RList(0);
        double R0_2 = R0 * R0;
        double v0 = state.vList(0);
        double v0_2 = v0 * v0;
        double v0_3 = v0_2 * v0;
        double v1 = state.vList(1);
        double v1_2 = v1 * v1;
        double v1_3 = v1_2 * v1;
        double Rho0 = state.RhoList(0);
        double Rho1 = state.RhoList(1);
        double Rho0_2 = Rho0 * Rho0;
        
        double bi_0 = bigIntInterp(v0);
        double bi_1 = bigIntInterp(v1);
        double smfp_0 = 600.0 * std::sqrt(M_PI) * v0 / params.sigma_0 / bi_0;
        double smfp_1 = 600.0 * std::sqrt(M_PI) * v1 / params.sigma_0 / bi_1;
        double lmfp_0 = 1.5 * params.a * params.c * Rho0 * v0_3 * params.sigma_0 * bi_0 / 512.0 ; 
        double lmfp_1 = 1.5 * params.a * params.c * Rho1 * v1_3 * params.sigma_0 * bi_1 / 512.0 ;

        state.LList(0) = - 2.0 / 3.0 * (state.uList(1) - state.uList(0)) / state.RList(1) * 
                       R0_2 * ( 
                        smfp_0 * lmfp_0 / (smfp_0 + lmfp_0) + smfp_1 * lmfp_1 / (smfp_1 + lmfp_1) 
                       );

        state.CList(0) = params.if_brem ? (params.brem_prefactor * Rho0_2 * v0 * bremIntInterp(v0)) : 0.0 ;
        
        for (int i = 1; i < (NoLayers-1); i++) {
            double Ri = state.RList(i);
            double Ri_2 = Ri * Ri;
            double Ri1 = state.RList(i+1);
            double Ri_1 = state.RList(i-1);
            
            double vi = state.vList(i);
            double vi_2 = vi * vi;
            double vi_3 = vi_2 * vi;
            
            double vi1 = state.vList(i+1);
            double vi1_2 = vi1 * vi1;
            double vi1_3 = vi1_2 * vi1;

            double Rhoi = state.RhoList(i);
            double Rhoi_1 = state.RhoList(i+1);
            double Rhoi_2 = Rhoi * Rhoi;

            double bii = bigIntInterp(vi);
            double bii_1 = bigIntInterp(vi1);
            double smfpi = 600.0 * std::sqrt(M_PI) * vi / params.sigma_0 / bii;
            double smfpi_1 = 600.0 * std::sqrt(M_PI) * vi1 / params.sigma_0 / bii_1;
            double lmfpi = 1.5 * params.a * params.c * Rhoi * vi_3 * params.sigma_0 * bii / 512.0 ; 
            double lmfpi_1 = 1.5 * params.a * params.c * Rhoi_1 * vi1_3 * params.sigma_0 * bii_1 / 512.0 ;

            state.LList(i) = - 2.0 / 3.0 * (state.uList(i+1) - state.uList(i)) / (Ri1 - Ri_1) * Ri_2 * ( 
                        smfpi * lmfpi / (smfpi + lmfpi) + smfpi_1 * lmfpi_1 / (smfpi_1 + lmfpi_1) 
                       );

            state.CList(i) = params.if_brem ? (params.brem_prefactor * Rhoi_2 * vi * bremIntInterp(vi)) : 0.0;
        }
    }
    
    void saveResults(std::ofstream& file, int tstep) {
        if (file.is_open()) {
            file << std::scientific << std::setprecision(10) << params.totalTime << " " << tstep << '\n'
                 << state.RList.transpose() << '\n'
                 << state.RhoList.transpose() << '\n'
                 << state.MList.transpose() << '\n'
                 << state.uList.transpose() << '\n'
                 << state.LList.transpose() << '\n'
                 << state.CList.transpose() << '\n';
        }
    }

    // ---------------------------------------------------------------------
    //  Precomputed big_int(vd) lookup: loading + interpolation
    // ---------------------------------------------------------------------

    // Load precomputed big_int(vd) lookup tables from files
    void loadBigIntTable() {
        // Filenames follow the same convention as other input lists
        std::string nameVdi = params.inputDir + "vdiList-" + params.tag + ".txt";
        std::string nameBi  = params.inputDir + "biList-"  + params.tag + ".txt";
        std::string nameBm  = params.inputDir + "bmList-"  + params.tag + ".txt";

        try {
            vd_grid = fileManager.readMatrix(nameVdi).array();
            bi_grid = fileManager.readMatrix(nameBi).array();
            bm_grid = fileManager.readMatrix(nameBm).array();
        } catch (const std::exception& e) {
            logger.error("Failed to read vdi/bi/bm lookup tables: " + std::string(e.what()));
            throw;
        }

        if (vd_grid.size() == 0 || bi_grid.size() == 0 || bm_grid.size() == 0) {
            throw std::runtime_error("vdi/bi lookup tables are empty.");
        }
        if (vd_grid.size() != bi_grid.size()) {
            std::ostringstream oss;
            oss << "Size mismatch between vdiList and biList: vdi size=" << vd_grid.size()
                << ", bi size=" << bi_grid.size();
            logger.error(oss.str());
            throw std::runtime_error("Size mismatch between vdiList and biList.");
        }
        if (vd_grid.size() != bm_grid.size()) {
            std::ostringstream oss;
            oss << "Size mismatch between vdiList and bmList: vdi size=" << vd_grid.size()
                << ", bm size=" << bm_grid.size() << "\n";
            logger.error(oss.str());
            throw std::runtime_error("Size mismatch between vdiList and bmList.");
        }

        logger.info("Loaded big_int lookup tables with " +
                    std::to_string(vd_grid.size()) + " points.");
    }

    // Simple log–log interpolation for big_int as a function of vd
    // vd is the dimensionless velocity in units of v_fid
    double bigIntInterp(double vd) const {
        if (vd_grid.size() == 0) {
            throw std::runtime_error("bigIntInterp called before vdi/bi tables were loaded.");
        }
        if (vd <= 0.0) {
            throw std::runtime_error("bigIntInterp: vd must be positive for log interpolation.");
        }

        // Assume table sorted ascending in vd
        if (vd <= vd_grid(0)) {
            return bi_grid(0);
        }
        int n = static_cast<int>(vd_grid.size());
        if (vd >= vd_grid(n - 1)) {
            return bi_grid(n - 1);
        }

        // Binary search for the interval [left, right] with
        // vd_grid[left] <= vd <= vd_grid[right]
        int left = 0;
        int right = n - 1;
        while (right - left > 1) {
            int mid = (left + right) / 2;
            if (vd_grid(mid) > vd) {
                right = mid;
            } else {
                left = mid;
            }
        }

        double x  = std::log(vd);
        double x0 = std::log(vd_grid(left));
        double x1 = std::log(vd_grid(right));
        double y0 = bi_grid(left);
        double y1 = bi_grid(right);

        // If values are not positive, fall back to linear interpolation
        if (y0 <= 0.0 || y1 <= 0.0) {
            double t_lin = (vd - vd_grid(left)) / (vd_grid(right) - vd_grid(left));
            return y0 + t_lin * (y1 - y0);
        }

        double ly0 = std::log(y0);
        double ly1 = std::log(y1);
        double t   = (x - x0) / (x1 - x0);
        double ly  = ly0 + t * (ly1 - ly0);
        return std::exp(ly);
    }

    // Simple log–log interpolation for brem_int as a function of vd
    // vd is the dimensionless velocity in units of v_fid
    double bremIntInterp(double vd) const {
        if (vd_grid.size() == 0) {
            throw std::runtime_error("bremIntInterp called before vdi/bi tables were loaded.");
        }
        if (vd <= 0.0) {
            throw std::runtime_error("bremIntInterp: vd must be positive for log interpolation.");
        }

        // Assume table sorted ascending in vd
        if (vd <= vd_grid(0)) {
            return bm_grid(0);
        }
        int n = static_cast<int>(vd_grid.size());
        if (vd >= vd_grid(n - 1)) {
            return bm_grid(n - 1);
        }

        // Binary search for the interval [left, right] with
        // vd_grid[left] <= vd <= vd_grid[right]
        int left = 0;
        int right = n - 1;
        while (right - left > 1) {
            int mid = (left + right) / 2;
            if (vd_grid(mid) > vd) {
                right = mid;
            } else {
                left = mid;
            }
        }

        double x  = std::log(vd);
        double x0 = std::log(vd_grid(left));
        double x1 = std::log(vd_grid(right));
        double y0 = bm_grid(left);
        double y1 = bm_grid(right);

        // If values are not positive, fall back to linear interpolation
        if (y0 <= 0.0 || y1 <= 0.0) {
            double t_lin = (vd - vd_grid(left)) / (vd_grid(right) - vd_grid(left));
            return y0 + t_lin * (y1 - y0);
        }

        double ly0 = std::log(y0);
        double ly1 = std::log(y1);
        double t   = (x - x0) / (x1 - x0);
        double ly  = ly0 + t * (ly1 - ly0);
        return std::exp(ly);
    }
};

// -------------------- Basic 解析：严格版（缺键就抛错） --------------------
static inline std::string ltrim(std::string s){
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch){return !std::isspace(ch);})); return s;
}
static inline std::string rtrim(std::string s){
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch){return !std::isspace(ch);} ).base(), s.end()); return s;
}
static inline std::string trim(std::string s){ return rtrim(ltrim(std::move(s))); }
static inline std::string lower(std::string s){
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){return std::tolower(c);}); return s;
}

static bool parse_bool01(std::string s){
    s = lower(trim(std::move(s)));
    if (s == "1" || s == "true" || s == "yes" || s == "on")  return true;
    if (s == "0" || s == "false"|| s == "no"  || s == "off") return false;
    throw std::runtime_error("Bad boolean (expect 0/1 or true/false): " + s);
}

static std::unordered_map<std::string,std::string>
parse_basic_kv(const std::string& path){
    std::unordered_map<std::string,std::string> kv; kv.reserve(32);
    std::ifstream in(path);
    if(!in) throw std::runtime_error("Cannot open Basic file: " + path);
    std::string line;
    while(std::getline(in, line)){
        // Strip comments starting with '#'
        auto pos_hash = line.find('#');
        if(pos_hash!=std::string::npos) {
            line = line.substr(0,pos_hash);
        }
        // Trim whitespace; skip empty/comment-only lines
        line = trim(std::move(line));
        if(line.empty()) continue;
        
        size_t pos = std::string::npos;
        size_t pos_eq = line.find('=');
        size_t pos_col= line.find(':');
        if(pos_eq!=std::string::npos) pos = pos_eq;
        else if(pos_col!=std::string::npos) pos = pos_col;
        std::string key, val;
        if(pos!=std::string::npos){
            key = trim(line.substr(0,pos));
            val = trim(line.substr(pos+1));
        } else {
            std::istringstream iss(line);
            if(!(iss>>key>>val)) continue;
            key = trim(key); val = trim(val);
        }
        if(key.empty() || val.empty()) continue;
        kv[lower(key)] = val;
    }
    return kv;
}

static void load_from_basic(const std::string& basic_path, SimulationParameters& P, Logger& logger){
    auto kv = parse_basic_kv(basic_path);
    auto gets = [&](const std::string& k)->std::optional<std::string>{
        auto it = kv.find(k); if(it==kv.end()) return std::nullopt; return it->second;
    };
    auto reqs = [&](const std::string& k)->std::string{
        auto s = gets(k); if(!s) throw std::runtime_error("Basic file missing required key: " + k); return *s;
    };
    auto getd = [&](const std::string& k)->std::optional<double>{
        auto s = gets(k); if(!s) return std::nullopt; try{ return std::stod(*s);}catch(...){
            throw std::runtime_error("Bad numeric value for key '"+k+"': "+*s);
        }
    };
    auto reqd = [&](const std::string& k)->double{
        auto v = getd(k); if(!v) throw std::runtime_error("Basic file missing required numeric key: " + k); return *v;
    };

    // required keys
    if (auto s = gets("name")) P.tag = *s;
    else if (auto s2 = gets("tag")) P.tag = *s2;
    else throw std::runtime_error("Basic file must provide 'name' (or 'tag').");

    P.a          = reqd("a");
    // P.b          = reqd("b");
    P.c          = reqd("c");
    P.sigma_0    = reqd("sigma_0");
    P.omega      = reqd("omega");
    P.brem_prefactor = reqd("brem_prefactor");
    P.mass_norm  = reqd("baryon_plummer_mass_norm");
    P.if_brem = parse_bool01(reqs("if_brem"));
    P.if_anni = parse_bool01(reqs("if_anni"));
    P.scale_norm = reqd("baryon_plummer_ars");
    P.age_of_universe = reqd("default_age_of_universe");
    P.epsilon    = reqd("epsilon");

    // derive paths from Basic location
    size_t slash = basic_path.find_last_of("/\\");
    std::string dir = (slash==std::string::npos) ? std::string(".") : basic_path.substr(0, slash);
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') dir.push_back('/');
    P.inputDir  = dir;
    P.outputFile= dir + "result-" + P.tag + ".txt";

    logger.info("Loaded Basic: " + basic_path);
    logger.debug("tag=" + P.tag + ", a=" + std::to_string(P.a) +
                 ", c=" + std::to_string(P.c) + ", sigma_0=" + std::to_string(P.sigma_0) + ", omega=" + std::to_string(P.omega) + 
                 ", mass_norm=" + std::to_string(P.mass_norm) + ", scale_norm=" + std::to_string(P.scale_norm) +
                 ", if_brem=" + std::string(P.if_brem ? "1" : "0") + ", if_anni=" + std::string(P.if_anni ? "1" : "0") +
                 (P.totalTime!=0.0 ? (", t="+std::to_string(P.totalTime)) : ""));
}

int main(int argc, char** argv) {
    clock_t start = clock();
    try {
        Logger logger(false);  // Set true for debug

        // Strict: must provide Basic path *prefix* and tag
        // Example: baseDir = "./test/initial/", tag = "20251104"
        // Basic file path becomes "./test/initial/20251104/Basic-20251104.txt"
        if (argc < 3) {
            logger.error("Usage: ./evolution <basic_dir_prefix> <tag>");
            logger.error("Example: ./evolution ./test/initial/ 20251104");
            return EXIT_FAILURE;
        }

        std::string baseDir = argv[1];
        std::string tag     = argv[2];
        if (!baseDir.empty() && baseDir.back() != '/' && baseDir.back() != '\\') {
            baseDir.push_back('/');
        }
        std::string basic_path = baseDir + tag + "/Basic-" + tag + ".txt";

        SimulationParameters params; // no defaults for physics; filled by Basic
        load_from_basic(basic_path, params, logger);

        FileManager fileManager(logger);

        // show loaded params (optional)
        params.display(logger);

        Simulator simulator(params, fileManager, logger);
        simulator.initialize();
        simulator.runSimulation();

        clock_t end = clock();
        float seconds = (float)(end - start) / CLOCKS_PER_SEC;
        logger.info("Computation time = " + std::to_string(seconds) + " s");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
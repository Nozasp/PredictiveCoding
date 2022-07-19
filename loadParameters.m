
function pars = loadParameters()

    % Runtime
    pars.verb  = false  ;
    pars.storeFields.v  = {'he', 'hi', 'hs'};
    pars.storeFields.dv = {};
    pars.storeFields.x  = {};

    % General parameters
    pars.N        = 20;
    pars.T        = 0.2:0.01:1; % periods of the generative oscillator (seconds)
    pars.f        = 1:pars.N; % unitless
    pars.dt       = 0.001; % seconds
    pars.initTime = 1;   % initalisation time (seconds)

    % Input paramters
    pars.In0    = 0;    % Spontaneous firing rate of input populations (Hz)
    pars.InMax  = 50;   % Max firing rate of input populations (Hz)
    pars.Iq0    = 0;    % Spontaneous firing rate of feedback populations (Hz)
    pars.IqMax  = 10;   % Max firing rate of feedback populations (Hz)

    % conductivities
    pars.Jee = 0.2;
    pars.Jie = 0.2; 
    pars.Jei = 1.4;
    pars.Jii = 6.7;
    pars.Jin = 0.008;
    pars.Jiq = 0.85;
    pars.Jes = 3.5;
    pars.Jsi = 0.12;
    pars.Jem = 2.2;

    % adaptation dynamics
    %pars.alpha    = 0; % adaptation strenght
    pars.alpha    = 0.022; % adaptation strenght
    pars.tauAdapt = 1.50;  % Adaptation time constant (s)

    % connectivity matrices
    pars.sigmaIn  = 3;
    pars.sigmaEI  = pars.sigmaIn;   
    pars.sigmaQie = pars.sigmaIn;
    pars.sigmaInh = [0.2,pars.sigmaIn];
    pars.wei  = gaussianFilter(pars.sigmaEI, pars.N);
    pars.wes  = eye(pars.N);
    pars.wie  = dogFilter(pars.sigmaInh(1), pars.sigmaInh(2), pars.N);
    pars.wii  = dogFilter(pars.sigmaInh(1), pars.sigmaInh(2), pars.N);
    %pars.wii  = 1-.5*gaussianFilter(1.5, pars.N);
    %pars.wii  = 0.7 * (ones(pars.N) - eye(pars.N)) + 0.3;
    % Whats the optimal bandwith optimising robustness and resolution?
    %plot(pars.f, [pars.wei(5, :); pars.wii(5, :); pars.wie(5, :)]); legend('wei', 'wii', 'wie');

    % time constants
    pars.ostoj   = false;
    pars.taue    = 0.020; % seconds [Brunel 2001] (4.5 +- 2.4 ms)
    pars.taui    = 0.010; % seconds [Brunel 2001] (4.5 +- 2.4 ms)
    pars.tauNMDA = 0.100; % NMDA-gating time constants (s) [Brunel 2001]
    pars.tauGABA = 0.005; % GABA-gating time constants (s) [Brunel 2001]
    pars.tauAMPA = 0.002; % AMPA-gating time constants (s) [Brunel 2001]

    % population parameters
    pars.gamma = 0.641;  % NMDA coupling   [Brunel 2001]
    pars.sigma = 0.0007; % noise amplitude [Wong 2006]
    pars.I0e   = 0.2346; % constant population input, exc (nA) [Wong2006]
    pars.I0i   = 0.17;   % constant population input, inh (nA)
    pars.ae    = 18.26;  
    pars.be    = -5.38;  
    pars.hme   = 78.67;   
    pars.ai    = 21.97;    
    pars.bi    = -4.81;    
    pars.hmi   = 125.62;  

end


function gaussW = gaussianFilter(s, N)

    k = 1:N;  
    n = 1 / (sqrt(2 * pi) * N * s);
    gaussW = n * exp(-(k-k').^2/(2 * s^2));
    gaussW = gaussW / (0.01^2 / max(gaussW(:)));

end



function dog = dogFilter(sIn, sOut, N)

    k = 1:N;  
    gaussIn  = exp(-(k-k').^2/(2 * sIn^2));
    gaussOut = exp(-(k-k').^2/(2 * sOut^2));

    dog = gaussOut-gaussIn;
    dog = dog / (0.88^2 / max(dog(:)));

end
function [u, In, Iq, timeSpace, f] = schnell(stim, pars)

    if nargin < 2, pars = loadParameters(); end
    if nargin < 1, stim = 'seq'; end
    
    if ischar(stim)
        stim = defaultTone(stim);
    end

    [In, Iq, f] = netInput(stim, pars);

    timeSpace = linspace(pars.dt, size(In, 2) * pars.dt, size(In, 2));
    
    u = solveSystem(timeSpace, In, Iq, pars);

    plotDynamics(u, In, Iq, timeSpace, f,   pars);

end



function u = solveSystem(timeSpace, In, Iq, pars)

    v = initaliseVariables(timeSpace, pars);
    u = [];

    for t = 2:length(timeSpace)
        [dv, x] = computeDerivatives(v, In(:, t), Iq(:, t), pars);
        v = updateVariables(v, dv, pars.dt);
        u = storeVariables(u, t, v, dv, x, pars.storeFields);
    end

end



function v = initaliseVariables(timeSpace, pars)

    x0 = zeros([pars.N, 1]);
    y0 = zeros([pars.N, length(pars.T)]);
    r0 = ones([pars.N, pars.N]);

    v = struct('he',  x0, 'hi',  x0, 'hs',  x0, 'hm', y0, 'dhm', y0, ...
               'rei', r0, 'res', r0, 'rii', r0, ...
               'Sa',  x0, 'Sg',  x0, 'Sn',  x0, 'Sm', x0);

    for t = 1:ceil(pars.initTime / pars.dt)
        dv = computeDerivatives(v, pars.In0, 0, pars);
        v  = updateVariables(v, dv, pars.dt);
    end

end



function v = updateVariables(v, dv, dt)

    F = fieldnames(v);
    for i = 1:length(F)
        vf = v.(F{i});
        vf = vf + dt * dv.(F{i});
        if not(strcmp(F{i}, 'hm') || strcmp(F{i}, 'dhm'))
            vf(vf < 0) = 0;
        end
        v.(F{i}) = vf;
    end

end



function u = storeVariables(u, t, v, dv, x, fields)

    F = fields.v;
    for i = 1:length(F)
        if isvector(v.(F{i}))
            u.(F{i})(:, t) = v.(F{i});
        elseif length(size(v.(F{i}))) == 2
            u.(F{i})(:, :, t) = v.(F{i});
        end
    end

    F = fields.x;
    for i = 1:length(F)
        if isvector(x.(F{i}))
        u.(F{i})(:, t) = x.(F{i});
        elseif length(size(x.(F{i}))) == 2
            u.(F{i})(:, :, t) = x.(F{i});
        end
    end

    F = fields.dv;
    for i = 1:length(F)
        if isvector(dv.(F{i}))
            u.(F{i})(:, t) = dv.(F{i});
        elseif length(size(dv.(F{i}))) == 2
            u.(F{i})(:, :, t) = dv.(F{i});
        end
    end

end



function [dv, x] = computeDerivatives(v, In, Iq, pars)

    % 1. Gating variables
    eta = pars.sigma * randn([pars.N, 4]);
    dv.Sa  = -v.Sa  / pars.tauAMPA + v.he + eta(:,1);
    dv.Sg  = -v.Sg  / pars.tauGABA + v.hi + eta(:,2);
    dv.Sn  = -v.Sn  / pars.tauNMDA + 0.641 * (1 - v.Sn) .* v.hs + eta(:,3);
    dv.Sm  = -v.Sm  / pars.tauAMPA + sum(v.hm, 2) + eta(:,4);

    % 2. Synaptic inputs
    x.he = pars.Jee * v.Sa - pars.Jie * pars.wie * v.Sg + pars.Jin * In;
    x.hi = pars.Jei * (pars.wei .* v.rei) * v.Sa - ...
           pars.Jii * (pars.wii .* v.rii) * v.Sg + pars.Jsi * v.Sn;
    x.hs = pars.Jes * (pars.wes .* v.res) * v.Sa + pars.Jiq * (Iq + v.Sm); 
    x.hm = pars.Jem * v.Sa;

    % 3. Plasticiy
    dv.rei = (1 - v.rei) / pars.tauAdapt + pars.alpha * v.hi * (v.he');
    dv.res = (1 - v.res) / pars.tauAdapt + pars.alpha * v.hs * (v.he');
    dv.rii = 0;% (1 - v.rii) / pars.tauAdapt + pars.alpha * v.hi * (v.hi');

    % 3. Rate variables
    dv.he = (- v.he + psix(x.he, 'e', pars)) ./ pars.taue;
    dv.hi = (- v.hi + psix(x.hi, 'i', pars)) ./ pars.taui;
    dv.hs = (- v.hs + psix(x.hs, 'e', pars)) ./ pars.taue;

    % 4. Generative oscillator
    T = repmat(pars.T, [pars.N, 1]);
    dv.dhm = - (2*pi ./ T).^2 .* v.hm;
    dv.hm  = -v.hm ./ T + psix(x.hm, 'e', pars) + v.dhm;

end



function px = psix(x, typ, pars)

    a   = pars.(['a',  typ]);
    b   = pars.(['b',  typ]);
    hm  = pars.(['hm',  typ]);

    px = hm  * (1 ./ (1 + exp(- (a * x + b))));

end



function [In, Iq, f] = netInput(stim, pars)

    if isempty(stim)
        stim = defaultStimulus();
    end

    % Instantaneous frequency
    f = zeros([stim.ISI/pars.dt, 1]);
    for ix = 1:length(stim.f)
        fx = stim.f{ix};
        f = [f; fx * [ones([stim.dur/pars.dt, 1]); zeros([stim.ISI/pars.dt, 1])]];
    end

    % Sensory input (bottom up)
    w = exp(-(pars.f - f).^2 / (2 * pars.sigmaIn^2));
    In = (f > 0)' .* pars.InMax .* w' + pars.In0;
    if isfield(stim, 'tail')
        In = [In, zeros([pars.N, stim.tail / pars.dt])];
    end

    % Top-down predictions
    if not(iscell(stim.pred))
        stim.pred = {};
        for i = 1:length(stim.f)
            stim.pred{i} = [0, 0, 0];
        end
    end

    w = zeros([round((stim.ISI + stim.predDt) / pars.dt), pars.N]);
    for ix = 1:length(stim.pred)
        px = stim.pred{ix};
        wx = 0;
        for jx = 1:size(stim.pred{ix}, 1)
            fx = px(jx, 1) * ones([round((stim.dur - stim.predDt)/pars.dt),1]);
            wx = wx + px(jx, 2) * exp(-(pars.f-fx).^2 / (2*(px(jx, 3)+eps)^2)); 
        end
        w = [w; wx; zeros([round((stim.ISI + stim.predDt)/pars.dt), pars.N])];
    end
    if stim.predDt < 0
        w = [w; zeros([- round(stim.predDt / pars.dt), pars.N])];
    else
        w = w(1:(size(w, 1) - round(stim.predDt/pars.dt)), :);
    end

    Iq = pars.IqMax .* w' + pars.Iq0;
    if isfield(stim, 'tail')
        Iq = [Iq, zeros([pars.N, stim.tail / pars.dt])];
    end

end


function plotDynamics(u, In, Iq, timeSpace, f, pars)

    subplot(511)
    imagesc(timeSpace, pars.f, In)
    ylabel('cf (unitless)')
    title('sensory input')
    %ylim(interval)
    xlim([0, max(timeSpace)])

    subplot(512)
    imagesc(timeSpace, pars.f, u.he)
    ylabel('cf (unitless)')
    title('he - excitatory')
    %ylim(interval)
    xlim([0, max(timeSpace)])

    subplot(513)
    imagesc(timeSpace, pars.f, u.hi)
    ylabel('cf (unitless)')
    title('hi - inhibitory')
    %ylim(interval)
    xlim([0, max(timeSpace)])

    subplot(514)
    imagesc(timeSpace, pars.f, u.hs)
    ylabel('cf (unitless)')
    title('hs - excitatory')
    %ylim(interval)
    xlim([0, max(timeSpace)])

    subplot(515)
    plot(timeSpace, sum(u.he))
    ylabel('cf (unitless)')
    title('mesoscopic')
    %ylim(interval)
    xlim([0, max(timeSpace)])

end



function stim = defaultTone(key)

    if strcmp(key, 'tone')
        stim.dur    = 0.2;
        stim.ISI    = 1;
        stim.f      = {8};
        stim.tail   = 0;
        stim.predDt = 0;
        stim.pred   = 0;
    elseif strcmp(key, 'miniseq')
        stim.dur    = 0.05;
        stim.ISI    = 0.75;
        stim.f      = {8, 8, 8, 8};
        stim.tail   = 0;
        stim.predDt = 0;
        stim.pred   = 0;
    elseif strcmp(key, 'seq')
        stim.dur    = 0.05;
        stim.ISI    = 0.75;
        stim.f      = {8, 8, 8, 12, 8, 8, 8};
        stim.tail   = 0;
        stim.predDt = 0;
        stim.pred   = 0;
    elseif strcmp(key, 'ctr')
        stim.dur    = 0.05;
        stim.ISI    = 0.75;
        stim.f      = {8, 12, 6, 10, 16, 4, 14};
        stim.tail   = 0;
        stim.predDt = 0;
        stim.pred   = 0;
    elseif strcmp(key, 'ctrLong')
        stim.dur    = 0.05;
        stim.ISI    = 0.75;
        stim.f      = num2cell(randi(12, [25, 1]) + 4);
        stim.tail   = 0;
        stim.predDt = 0;
        stim.pred   = 0;
    elseif strcmp(key, 'seqpred')
        stim.dur    = 0.05;
        stim.ISI    = 0.75;
        stim.predDt = -0.1;
        stim.f      = {8, 8, 8, 12, 8, 8, 8};
        stim.tail   = 0;
        stim.pred   = {[8, 0, 0], ... % [core, amplitude, width]
                       [8, 1, 0], ...
                       [8, 1, 0], ...
                       [8, 1, 0], ...
                       [8, 1, 0], ...
                       [8, 1, 0], ...
                       [8, 1, 0]};
    end

end
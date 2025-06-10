%%% Generate ABM data for BDM model

clear
clc

rp_vec = 0.01:0.01:5;
V = 1;         % compartment size
m = 0.05;      % initial condition, i.e. initial agent density
real_vec = 25; % total number of ABM simulations/runs

for i = 1:length(rp_vec)

    disp(i)
    rp = rp_vec(i);
    rd_vec = rp/2;

    for j = 1:length(rd_vec)

        rm = 1;
        rd = rd_vec(j);

        disp([rp,rd])

        [ABM_sim_all, U, t, F_all] = ...
            Prolif_assay_func(V,rm,rp,rd,20/(rp-rd),real_vec(end),m); % run BDM model

        dt = t(2) - t(1);  % time step

        Ut = smoothdata(Finite_diff_1d(U,dt))'; % smooth derivatives

        %%% Only save mean dataset
        for real = real_vec

            ABM_sim = mean(ABM_sim_all(:,1:real),2);
            ABM_sim_t = smoothdata(Finite_diff_1d(ABM_sim,dt));
            F = mean(F_all(:,1:real),2);

            save(['../../data/ABM/logistic_ABM_sim_rp_' num2str(rp) '_rd_' num2str(rd)...
                '_rm_' num2str(rm) '_m_' num2str(m) '_real' num2str(real) '.mat']...
                ,'ABM_sim','m','V','rd','rp','t','U','rm','Ut','ABM_sim_t','F')

        end

    end


end


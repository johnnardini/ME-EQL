function Ut = Finite_diff_1d(U,dt)

    Ut = zeros(size(U));
    
    %compute Ut
    Ut(2:end-1) = (U(3:end) - U(1:end-2))/(2*dt);
    Ut(1) = (U(2) - U(1))/(dt);
    Ut(end) = (U(end) - U(end-1))/(dt);
    
    
end
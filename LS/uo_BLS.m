% bls function 
function [al, iWout] = uo_BLS(x, d, f, g, almax, almin, rho, c1, c2, iW)
    al = almax;
    iWout = 0;

    WC1  = @(al) f(x + al * d) <= f(x) + c1 * al * g(x)' * d;
    WC2  = @(al) g(x + al * d)' * d >= c2 * g(x)' * d;
    SWC2 = @(al) abs(g(x + al * d)' * d) <= c2 * abs(g(x)' * d);

    while al > almin
        % if iW = 1, we only check WC
        if iW == 1
            % first we check both WC1 and WC2, if both are satisfied, iWout = 2
            if WC1(al) & WC2(al)
                iWout = 2;
                break;
            % if WC1 is satisfied but not WC2, iWout = 1
            elseif WC1(al)
                iWout = 1;
                break;
            end
        % if iW = 2, we check WC1 and SWC2
        elseif iW == 2
            % we only check WC1 and SWC2, if both are satisfied, iWout = 3
            if WC1(al) & SWC2(al)
                iWout = 3;
                break;
            end
        end
        % update alpha
        al = al * rho;
    end
    if al <= almin
        al = almin;
        if WC1(al)
            iWout = 1;
        else 
            iWout = 0;
        end
    end
end

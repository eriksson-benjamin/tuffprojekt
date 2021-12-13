

% upprepa experimentet 500 gånger
means = zeros(1, 500);
for i=1:length(means)
    x = normrnd(0, 1, 1, 1000); % generera 1000 mätvärden x_i med mu=0 och sigma=1
    means(i) = mean(x); % beräkna medelvärdet
end

x = normrnd(0, 1, 1, 1000); % generera 1000 mätvärden x_i med mu=0 och sigma=1
m = mean(x); % medelvärde
s = std(x); % standardavvikelse
ms = s/sqrt(length(x)); % standardosäkerhet i medelvärdet
mean_std = std(means); % spridningen i medelvärdet från många experiment

disp('sanna värden: mu = 0, sigma = 1')
disp(sprintf('medelvärde = %f', m))
disp(sprintf('standardavvikelse = %f', s))
disp(sprintf('standardavvikelse i medelvärde = %f', ms))
disp(sprintf('spridning i medelvärde = %f', mean_std))

% x = normrnd(1, 0.1, 1, 5)
x = [0.97, 1.09, 1.16, 1.04, 1.12]
m = mean(x)
s = std(x)
sm = s/sqrt(length(x))

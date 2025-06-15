%% FUNCTIONS
% Standard l2 fourier transform function
function c = l2(x, phi)
    c = phi' * x';
end

% l1 fourier transform function
function c = l1(x, phi, iteration, epsilon)
    if nargin < 3 || isempty(iteration)
        iteration = 60; % Default
    end

    if nargin < 4 || isempty(epsilon)
        epsilon = 1.1e-4; % Default
    end
    
    xM = x + 5;
    [~, m] = size(phi);
    c = zeros(m, 1);

    for i=(1:iteration)
        e = abs(x - xM);
        e = max(e, epsilon);
        lambda = diag(1 ./ e); % Weigths matrix
    
        % Coefficients update
        c_new = (phi' * (lambda * phi)) \ (phi' * (lambda * x'));
    
        
        if norm(c_new - c) < epsilon
            fprintf('Convergence reached at iteration %d\n', i);
            break;
        end
        
        c = c_new;
        xM = real(phi * c)';
    end
    c = c_new;
end
%% PARAMETERS
N = 2000; % Signal length
M = 32;   % Number of coefficients

range = floor(M/2);

k = (-range : range);
phi = exp(1j * 2 * pi * k' * (1:N) / N)' * 1/sqrt(N);


%% Step function for Gibbs phenomenon   (N=1000, M=128)
y = linspace(0, 100, N);
x = (y >= 22.5 & y < 77.5);
%plot(x)
%% Step function for stability of l1    (N=1000, M=24)
y = linspace(0, 100, N);
x = (y >= 48 & y < 51);
%plot(x)
%% STANDARD ECG                         (N=2000, M=32)
load("ECG.mat");
x = ECG(100:N+99)';
%Sampling rate 500 => with N=2000 -> M=32 => 8hz low pass filter
%% ADD SPIKE NOISE TO ECG
impulse_noise = zeros(size(ecg));
impulse_noise(randi(length(ecg), [1, 20])) = 0.5;  %add 20 spike randomly
x = ecg + impulse_noise;

plot(x)
%% Ventricular hypertrophy ECG          (N=2000, M=32)
[record, fs] = rdsamp('02412_hr', 1);
%Sampling rate 500 => with N=2000 -> M=32 => 8hz low pass filter
%https://physionet.org/content/ptb-xl/1.0.1/

x = record(1000:999+N)';
x = x - mean(x);
plot(x)

%% Atrial fibrillation                  (N=2000, M=64)
[record, fs] = rdsamp('08378', 1);
%Sampling rate 250 => with N=2000, M=64 => 8hz low pass filter
%https://physionet.org/content/afdb/1.0.0/

x = record(63000:62999+N)';
x = x - mean(x);
plot(x)


%% Compute L2 and L1 and Zero-fase Butterworth
c_new = l1(x, phi);
x_l1 = real(phi * c_new);
c_new = l2(x, phi);
x_l2 = real(phi * c_new);

avgl2 = mean(abs(x-x_l2'));
avgl1 = mean(abs(x-x_l1'));

Fc = 8;
Wn = Fc / (fs / 2);
[b, a] = butter(1, Wn);
but = filtfilt(b, a, x);

avgB = mean(abs(x-but));

%% Plot the filtering with l2 | l1 | Zero-fase Butterworth
subplot(3, 1, 1)
residual = x - x_l2';
plot(x, LineWidth=1.5); hold on;
plot(x_l2', 'Red', LineWidth=2); hold on;
plot(residual, 'Green', LineWidth=2);

subplot(3, 1, 2)
residual = x - x_l1';
plot(x, LineWidth=1.5); hold on;
plot(x_l1', 'Red', LineWidth=2); hold on;
plot(residual, 'Green', LineWidth=2);

subplot(3, 1, 3)
residual = x - but;
plot(x, LineWidth=1.5); hold on;
plot(but, 'red', LineWidth=2); hold on;
plot(residual, 'Green', LineWidth=2);
fprintf('MAE\nl2: %.4f\nl1: %.4f\nBut: %.4f\n', avgl2, avgl1, avgB);

%% L2 and L1 in the same plot
plot(x, LineWidth=1.5); hold on;
plot(x_l2', LineWidth=2);
hold on;
plot(x_l1', 'black', LineWidth=2);
fprintf('MAE\nl2: %.4f\nl1: %.4f\n', avgl2, avgl1);

%% Plot L2 only
plot(x, LineWidth=1.5); hold on;
plot(x_l2', LineWidth=2);
%xlabel(sprintf('ℓ2 error = %d', avgl2) );
fprintf('MAE\nl2: %.4f\n', avgl2);

%% Plot L1 only
plot(x, LineWidth=1.5); hold on;
plot(x_l1', 'black', LineWidth=2);
%xlabel(sprintf('ℓ1 error = %d', avgl1) );
fprintf('MAE\nl1: %.4f\n', avgl1);
function [stats] = PlotRoiDistanceAllRoiDurationPeriod(data, veThresh, showPlots, binSteps, meanThresh)
%Makes various plots from the output of RoiDistanceRatio. THis version is
%made to use Log2Lin models. It's a simpler version of
%PlotRoiDistanceDataAllBin, giving fewer and simpler outputs

% showPlots options
% 1 = 
% 2 =
% 3 =
% 4 =
% 5 = Plot of Tuning Functions
% 6 =
% 7 =

if ~exist('veThresh', 'var') || isempty(veThresh)
    veThresh=0;
end
if ~exist('showPlots', 'var') || isempty(showPlots)
    showPlots=[1 1 1 1 1 1];
end
if ~exist('binSteps', 'var') || isempty(binSteps)
    binSteps=11;
end
if ~exist('meanThresh', 'var') || isempty(meanThresh)
    meanThresh=zeros(size(data,1), 1);
end

colors={'k', 'r', 'g', 'b', 'm', 'c'};

dataAll=data;
clear data;
try
    figure;
for whichRoi=1:size(dataAll, 1)
    if ~isempty(dataAll{whichRoi,1})
        data{1}=dataAll{whichRoi,1};
        data{2}=dataAll{whichRoi,2};
        data{3}=dataAll{whichRoi,3};
        if length(data{1}.meanDist)>1
            data{1}.meanDist = mean(data{1}.meanDist);
            data{2}.meanDist = mean(data{2}.meanDist);
            data{3}.meanDist = mean(data{3}.meanDist);
        end
        
        %for log2lin of sigmas
        try
        for n=1:length(data)
            %         %Just run this part the first time through
            %
            %         data{n}.x0sLog=data{n}.x0s;
            %         data{n}.sigmasLog=data{n}.sigmas;
            %         fwhms=data{n}.sigmas.*(2*sqrt(2*log(2)));
            %         data{n}.sigmas=exp(data{n}.x0s+fwhms./2)-exp(data{n}.x0s-fwhms./2);
            %         data{n}.x0s=exp(data{n}.x0s);
            %
            %n=1;
            data{n}.veIndices=data{n}.ves>=veThresh & data{n}.x0s>=0.05 & data{n}.x0s<=1 & data{n}.y0s>=0.05 & data{n}.y0s<=1; % & data{1}.meanSignal>=meanThresh(whichRoi);
            dataAll{whichRoi, n}.veIndices=data{n}.ves>=veThresh & data{n}.x0s>=0.05 & data{n}.x0s<=1 & data{n}.y0s>=0.05 & data{n}.y0s<=1; %& data{1}.meanSignal>=meanThresh(whichRoi);
            %         %data{n}.x0fit=linreg(data{n}.ratio(data{n}.veIndices), data{n}.x0s(data{n}.veIndices), data{n}.ves(data{n}.veIndices));
            %         %data{n}.sigmafit=linreg(data{n}.ratio(data{n}.veIndices), data{n}.sigmas(data{n}.veIndices), data{n}.ves(data{n}.veIndices));
        end
        
        
        
       
        b=linspace(0.5/binSteps, 1-(0.5/binSteps), binSteps);
        %number vs mean distance
        %xpoints=0:0.01:data{1}.meanDist;
        %if showPlots(1)==1 || showPlots(2)==1

%         bins=(1+first:stepsize:ceil(data{n}.meanDist)-last)./data{n}.meanDist;
%         bins=(1+first:stepsize./data{n}.meanDist:ceil(data{n}.meanDist)-last)./data{n}.meanDist;
%         bins=(1+first:(ceil(data{n}.meanDist)-last)./ss:ceil(data{n}.meanDist)-last)./data{n}.meanDist;
%         bins=(1+first:data{n}.meanDist/ss:ceil(data{n}.meanDist)-last)./data{n}.meanDist;
%         bins=(1:2:ceil(data{n}.meanDist))./data{n}.meanDist;
        bins=(1:2:ceil(data{n}.meanDist))./data{n}.meanDist;
        start = 1; % length of ROI
        ending = ceil(data{n}.meanDist);
        numsteps =  ceil(ending/1.77); %1.7 millimeter scanning resolution
        
        bins = linspace(start,ending,numsteps)./data{n}.meanDist;
        catch
           fprintf('indexing data based on thresholds (until 78)')
        end
        
        for n=1:length(data)
            if sum(data{n}.veIndices)>1
                tmp=corrcoef(data{n}.ratio(data{n}.veIndices),data{n}.x0s(data{n}.veIndices));
                rDistDurCorr(whichRoi, n)=tmp(1,2);
                tmp=corrcoef(data{n}.ratio(data{n}.veIndices),data{n}.y0s(data{n}.veIndices));
                rDistPerCorr(whichRoi, n)=tmp(1,2);
                ns(whichRoi, n)=floor(sum(data{n}.veIndices)./1.77^2);
                pDistDurCorr(whichRoi, n)=r2p(rDistDurCorr(whichRoi, n), ns(whichRoi, n));
                pDistPerCorr(whichRoi, n)=r2p(rDistPerCorr(whichRoi, n), ns(whichRoi, n));
                
                tmp=corrcoef(data{n}.x0s(data{n}.veIndices), data{n}.y0s(data{n}.veIndices));
                rDurPerCorr(whichRoi,n)=tmp(1,2);
                pDurPerCorr(whichRoi, n)=r2p(rDurPerCorr(whichRoi, n), ns(whichRoi, n));
            end
            %     %Quick and dirty way, evenly-spaced bins at odd intervals
            %     data{n}.x{4}=b.*data{n}.meanDist;
            %     data{n}.y{4}=data{n}.y{3};
            %     data{n}.ysterr{4}=data{n}.ysterr{3};
            %     [data{n}.ypoints, data{n}.logLineFit, data{n}.bupper, data{n}.blower]=bootstrapLogLineFitter(data{n}.x{4},data{n}.y{4},1./data{n}.ysterr{4}(1,:), xpoints);
            
            %   Slower, but with bins at regular distance intervals
            if showPlots(1)==1 || showPlots(2)==1
                data{n}.y{4}=nan(size(bins));
                data{n}.ysterr{4}=nan(2,length(bins));
                data{n}.y{5}=nan(size(bins));
                data{n}.ysterr{5}=nan(2,length(bins));
                for bCount=1:length(bins)
                    bii = data{n}.ratio> bins(bCount)-bins(1) & ...
                        data{n}.ratio< bins(bCount)+bins(1) & data{n}.veIndices;
                    if any(bii)
                        s=wstat(data{n}.x0s(bii), data{n}.ves(bii), 1.77^2);
                        data{n}.y{4}(bCount)=s.mean;
                        data{n}.ysterr{4}(:,bCount)=s.sterr;
                        s=wstat(data{n}.y0s(bii), data{n}.ves(bii), 1.77^2);
                        data{n}.y{5}(bCount)=s.mean;
                        data{n}.ysterr{5}(:,bCount)=s.sterr;
                    end
                end
                
                
                try
                data{n}.x{4}=bins;
                data{n}.x{5}=bins;
%                 tmp4=[data{n}.y{4} data{n}.y{4} data{n}.y{4}];
%                 tmp5=[data{n}.y{5} data{n}.y{5} data{n}.y{5}];
% %                 tmpbins=(1:2:(3*ceil(data{n}.meanDist)))./data{n}.meanDist;
%                 bigending = 3*ceil(data{n}.meanDist);
%                 bigsteps = ceil(bigending/1.7);
%                 tmpbins=linspace(1,bigending,bigsteps)./data{n}.meanDist;
%                 try
%                     tmp4r=resample(tmp4, tmpbins);
%                     tmp5r=resample(tmp5, tmpbins);
%                 catch
%                     tmp4r=resample(tmp4(1:end-1), tmpbins);
%                     tmp5r=resample(tmp5(1:end-1), tmpbins);
%                 end
%                 data{n}.y{4}=tmp4r(length(data{n}.y{4})+1:2*length(data{n}.y{4}));
%                 data{n}.y{5}=tmp5r(length(data{n}.y{5})+1:2*length(data{n}.y{5}));
                xpoints=min(bins):0.01:max(bins);
%                 xpoints = min(bins(~isnan(data{n}.y{4}))):1/(ss+1):max(bins(~isnan(data{n}.y{4})));
%                 xpoints = min(bins(~isnan(data{n}.y{4}))):1/(ss+1):max(bins(~isnan(data{n}.y{4})));
% %                 data{n}.xpoints=xpoints;
                % Uses the full set of ratio and x0s, hence, reducing bins
                % does not help to reduce the
                [data{n}.ypointsXlinear, data{n}.LineFitXlinear,...
                    data{n}.bupperXlinear, data{n}.blowerXlinear,...
                    data{n}.Blinear]=bootstrapLineFitter(data{n}.ratio(data{n}.veIndices),...
                    data{n}.x0s(data{n}.veIndices),data{n}.ves(data{n}.veIndices), bins);
                [data{n}.ypointsYlinear, data{n}.LineFitYlinear,...
                    data{n}.bupperYlinear, data{n}.blowerYlinear,...
                    data{n}.Blinear]=bootstrapLineFitter(data{n}.ratio(data{n}.veIndices),...
                    data{n}.y0s(data{n}.veIndices),data{n}.ves(data{n}.veIndices), bins);
                
                linFitX=polyfit(data{n}.ratio(data{n}.veIndices),data{n}.x0s(data{n}.veIndices), 1);
                flatDurX=data{n}.x0s(data{n}.veIndices)-polyval(linFitX, data{n}.ratio(data{n}.veIndices));
%                 flatDurX2=data{n}.x0s(data{n}.veIndices)-polyval(linFitX, data{n}.x0s(data{n}.veIndices));
%                 figure; plot(flatDurX); hold on; plot(flatDurX2)
                
                linFitY=polyfit(data{n}.ratio(data{n}.veIndices),data{n}.y0s(data{n}.veIndices), 1);
                flatDurY=data{n}.y0s(data{n}.veIndices)-polyval(linFitY, data{n}.ratio(data{n}.veIndices));
                
                % Duration for spectral analysis
%                 linFitXforSpec = polyfit(data{n}.xpoints(~isnan(data{n}.y{4})),data{n}.y{4}(~isnan(data{n}.y{4})), 1);
%  flatDurXforSpec = data{n}.y{4}(~isnan(data{n}.y{4}))-polyval(linFitXforSpec,data{n}.xpoints(~isnan(data{n}.y{4})));
%                 linFitXforSpec = polyfit(bins,data{n}.y{4}, 1);
%                 flatDurXforSpec = data{n}.y{4}(~isnan(data{n}.y{4}))-polyval(linFitXforSpec,bins);
%                 % Period for spectral analysis
%                 linFitYforSpec = polyfit(bins,data{n}.y{5}(~isnan(data{n}.y{5})), 1);
%                 flatDurYforSpec = data{n}.y{5}(~isnan(data{n}.y{5}))-polyval(linFitYforSpec,bins);
                catch
                    fprintf('linear fit, 121-148.\n')
                end
                
                try
%                 % Setup fft
% %                 L=length(data{n}.xpoints(~isnan(data{n}.y{4})));
%                 L=length(bins);
%                 h = hanning(L).^(1/10);
%                 np = ceil(log2(L));
%                 f = (data{n}.meanDist/2^np)*(0:2^(np-1)-1);
%                 
% %                 tmpd = flatDurXforSpec.*h';
% %                 tmpp = flatDurYforSpec.*h';
%                 tmpd = flatDurXforSpec;
% %                 tmpd =data{n}.y{4}(~isnan(data{n}.y{4}));
% %                 tmpp = data{n}.y{5}(~isnan(data{n}.y{5}));
%                 tmpp = flatDurYforSpec;
%                 % Alternative statistic for num cycles per map:
%                 fftd = fft(tmpd, 2^np)/(L/2);
%                 fftp = fft(tmpp, 2^np)/(L/2);
% %                 tmpd = fft(flatDurXforSpec, 2^np)/(L/2);
% %                 tmpp = fft(flatDurYforSpec, 2^np)/(L/2);
% %                 fftd = tmpd(end:-1:round(length(tmpd)/2))/(L/2);
% %                 fftp = tmpp(end:-1:round(length(tmpp)/2))/(L/2);
%                 stats.DurationSpectrum{whichRoi} = abs(fftd(1:2^(np-1)));
%                 stats.PeriodSpectrum{whichRoi} = abs(fftp(1:2^(np-1)));
%                 stats.fft{whichRoi}.vectorlength = L;
%                 stats.fft{whichRoi}.hwin = h;
%                 stats.fft{whichRoi}.bitsize = np;
%                 stats.fft{whichRoi}.fpoints = f;
                stats.bins{whichRoi} = length(bins);
                
%                 figure, plot(f,abs(fftd(1:2^(np-1))));
%                 figure, plot(f,abs(fftp(1:2^(np-1))));
                
                catch
                    fprintf('fft, 151-179.\n')
                end


                try
%                     [data{n}.ypointsX, data{n}.logLineFitX, data{n}.bupperX, data{n}.blowerX, data{n}.B]=bootstrapLogLineFitter(data{n}.x{4},data{n}.y{4},1./data{n}.ysterr{4}(1,:), xpoints);
%                     [data{n}.ypointsY, data{n}.logLineFitY, data{n}.bupperY, data{n}.blowerY, data{n}.B]=bootstrapLogLineFitter(data{n}.x{5},data{n}.y{5},1./data{n}.ysterr{5}(1,:), xpoints);
% data{n}.x{4} duration % {5} period is binned data
%                     [data{n}.ypointsX, data{n}.logLineFitX, data{n}.bupperX, data{n}.blowerX, data{n}.B]=bootstrapLogLineFitter(data{n}.ratio(data{n}.veIndices),flatDurX,data{n}.ves(data{n}.veIndices), xpoints);
%                     [data{n}.ypointsY, data{n}.logLineFitY, data{n}.bupperY, data{n}.blowerY, data{n}.B]=bootstrapLogLineFitter(data{n}.ratio(data{n}.veIndices),flatDurY,data{n}.ves(data{n}.veIndices), xpoints);
                    [data{n}.ypointsX, data{n}.logLineFitX, data{n}.bupperX, data{n}.blowerX, data{n}.B]=bootstrapLogLineFitter(data{n}.ratio(data{n}.veIndices),flatDurX,data{n}.ves(data{n}.veIndices), bins);
                    [data{n}.ypointsY, data{n}.logLineFitY, data{n}.bupperY, data{n}.blowerY, data{n}.B]=bootstrapLogLineFitter(data{n}.ratio(data{n}.veIndices),flatDurY,data{n}.ves(data{n}.veIndices), bins);
                    
                    if n == 1
%                         newvec = [0.1 : 0.1 : max(data{n}.logLineFitX(1),data{n}.logLineFitY(1))*3];

                        % start at half cycle, in steps of quarter cycle
                        % step until nyum cycles == nyquist frequency
                        newvec = [pi : pi/2 : (length(data{n}.y{4})/2)*pi*2]; 
%                         newvec2 = [0.1 : 0.1 : data{n}.logLineFitY(1)*5];
%                         newvec = [data{n}.logLineFitX(1)/2 : 0.01 : data{n}.logLineFitX(1)/2+data{n}.logLineFitX(1)];
                        compline = linspace(0.5/length(data{n}.y{4}),1-0.5/length(data{n}.y{4}),length(data{n}.y{4}));
                        clear newfitd newfitp sfitd sfitp coherencesd coherencesp sincoherencesd sincoherencesp
                        for nf = 1:length(newvec)
                            % Find correlation between predicted with
                            % different frequencies and the real progression
                            newfitd(nf,:) = data{n}.logLineFitX(3).*sin(newvec(nf).*compline+data{n}.logLineFitX(2))+polyval(linFitX, compline);
                            sfitd(nf,:) = data{n}.logLineFitX(3).*sin(newvec(nf).*compline+data{n}.logLineFitX(2));
                            rd = corrcoef(newfitd(nf,:),data{n}.y{4}, 'Rows', 'complete');
                            rd2 = corrcoef(sfitd(nf,:),data{n}.y{4}, 'Rows', 'complete');
                            coherencesd(nf) = rd(1,2);
                            sincoherencesd(nf) = rd2(1,2);
                            % Same for Period
                            newfitp(nf,:) = data{n}.logLineFitY(3).*sin(newvec(nf).*compline+data{n}.logLineFitY(2))+polyval(linFitY, compline);
                            sfitp(nf,:) = data{n}.logLineFitY(3).*sin(newvec(nf).*compline+data{n}.logLineFitY(2));
                            rp = corrcoef(newfitp(nf,:),data{n}.y{5}, 'Rows', 'complete');
                            rp2 = corrcoef(sfitp(nf,:),data{n}.y{5}, 'Rows', 'complete');
                            coherencesp(nf) = rp(1,2);
                            sincoherencesp(nf) = rp2(1,2);
                        end
                        stats.coherencesDur{whichRoi} = coherencesd;
                        stats.sinCoherenceDur{whichRoi} = sincoherencesd;
                        stats.coherencesPer{whichRoi} = coherencesp;
                        stats.sinCoherencePer{whichRoi} = sincoherencesp;
                        stats.coherenceAxis{whichRoi} = newvec;%./data{n}.meanDist);
                        stats.meanDist{whichRoi} = data{n}.meanDist;
                        rLinDur = corrcoef(polyval(linFitX,compline),data{n}.y{4});
                        rLinPer = corrcoef(polyval(linFitY,compline),data{n}.y{5});
                        stats.rLinFitX{whichRoi} = rLinDur(1,2);
                        stats.rLinFitY{whichRoi} = rLinPer(1,2);
                    end
                    
                    % Duration
                    % GLM for different approach to fit per component
                    sind = find(coherencesd==max(coherencesd)); % Find index of best fitting cyclical fit
                    sineBins = sfitd(sind,:);
                    linBins = compline; % Make linear fit
                    dataBins = data{n}.y{4}';
                    tmp = ~isnan(dataBins);
                    sineBins = sineBins(tmp);
                    linBins = linBins(tmp);
                    dataBins = dataBins(tmp);
                    
                    X=[linBins' sineBins' ones(size(linBins))'];
                    Y=dataBins; % Actual data
                    B_hat=pinv(X)*Y; %scaling terms
                    U=Y-X*B_hat; % Residuals
                    df = size(Y,1) - size(X,2); % Degrees of freedom
                    for whichPredictor=1:2
                        C=[0 0 0];
                        C(whichPredictor)=1;
                        SE = sqrt((sum(U.^2)./df)*(C*pinv(X'*X)*C'));
                        T(whichPredictor)=C*B_hat./SE;
                    end
                    
                    glm_pvals = t2p(abs(T),1,df);
                    sin_pvals = glm_pvals(1);
                    lin_pvals = glm_pvals(2);
                    
                    % Period
                    sinp = find(coherencesp==max(coherencesp)); % Find index of best fitting cyclical fit
                    sineBins = sfitp(sinp,:);
                    linBins = compline; % Make linear fit
                    dataBins = data{n}.y{5}';
                    tmp = ~isnan(dataBins);
                    sineBins = sineBins(tmp);
                    linBins = linBins(tmp);
                    dataBins = dataBins(tmp);
                                        
                    Xp=[linBins' sineBins' ones(size(linBins))'];
                    Yp=dataBins; % Actual data
                    B_hatp=pinv(Xp)*Yp; %scaling terms
                    Up=Yp-Xp*B_hatp; % Residuals
                    dfp = size(Yp,1) - size(Xp,2); % Degrees of freedom
                    for whichPredictor=1:2
                        C=[0 0 0];
                        C(whichPredictor)=1;
                        SEp = sqrt((sum(Up.^2)./dfp)*(C*pinv(Xp'*Xp)*C'));
                        Tp(whichPredictor)=C*B_hatp./SEp;
                    end
                    
                    glmp_pvals = t2p(abs(Tp),1,dfp);
                    sinp_pvals = glmp_pvals(1);
                    linp_pvals = glmp_pvals(2);
                    
                    
                    %% 95% CI's
                    %  data{n}.ypointsX = data{n}.ypointsX+polyval(linFitX, xpoints);
%                     data{n}.ypointsY = data{n}.ypointsY+polyval(linFitY, xpoints);
%                     data{n}.bupperX = data{n}.bupperX+polyval(linFitX, xpoints)';
%                     data{n}.bupperY = data{n}.bupperY+polyval(linFitY, xpoints)';
%                     data{n}.blowerX = data{n}.blowerX+polyval(linFitX, xpoints)';
%                     data{n}.blowerY = data{n}.blowerY+polyval(linFitY, xpoints)';
                    data{n}.ypointsX = data{n}.ypointsX+polyval(linFitX, bins);
                    data{n}.ypointsY = data{n}.ypointsY+polyval(linFitY, bins);
                    
                    % bupperX =  constant on the 
                    data{n}.bupperX = data{n}.bupperX+polyval(linFitX, bins)';
                    data{n}.bupperY = data{n}.bupperY+polyval(linFitY, bins)';
                    data{n}.blowerX = data{n}.blowerX+polyval(linFitX, bins)';
                    data{n}.blowerY = data{n}.blowerY+polyval(linFitY, bins)';
                catch
                    n
                    fprintf('Fitting.\n')
                end
                if n == 1
                    stats.linFitDur(whichRoi,:) = linFitX;
                    stats.linFitPer(whichRoi,:) = linFitY;
                    stats.sinFitDur(whichRoi,:) = data{n}.logLineFitX;
                    stats.sinFitPer(whichRoi,:) = data{n}.logLineFitY;
                    
                    % Duration
                    stats.sin_pvals{whichRoi} = sin_pvals;
                    stats.lin_pvals{whichRoi} = lin_pvals;
                    stats.Bfit{whichRoi} = B_hat;
                    stats.Yhat{whichRoi} = X*B_hat;
                    stats.residuals{whichRoi} = U;
                    stats.df{whichRoi} = df;
                    stats.Tvals{whichRoi} = T;
                    stats.SE = SE;
                    
                    % Period
                    stats.sinp_pvals{whichRoi} = sinp_pvals;
                    stats.linp_pvals{whichRoi} = linp_pvals;
                    stats.Bfitp{whichRoi} = B_hatp;
                    stats.Yhatp{whichRoi} = Xp*B_hatp;
                    stats.residualsp{whichRoi} = Up;
                    stats.dfp{whichRoi} = dfp;
                    stats.Tvalsp{whichRoi} = Tp;
                    stats.SEp = SEp;
                    
                end
                %Statistical test of progression with distance
                if showPlots(1)==1
                    permutations=10000;
                    ydat=log(data{n}.y{4}(~isnan(data{n}.y{4})));
                    xdat=data{n}.x{4}(~isnan(data{n}.y{4}));
                    fitDist=zeros(2,permutations);
                    for m=1:permutations
                        yshuffle=ydat(randperm(length(ydat)));
                        fitDist(:,m)=linreg(xdat, yshuffle);
                    end
                    CI=[prctile(fitDist(2,:), 2.5) prctile(fitDist(2,:), 97.5)];
                    max(fitDist(2,:));
                    measure=linreg(xdat, ydat);
                    measure(2);
                    [whichRoi, n]
                    if measure(2)<max(fitDist(2,:))
                        tmp= fitDist(2,:);
                        tmp=sort(tmp);
                        tmp=tmp>measure(2);
                        pXprog(whichRoi, n)=(permutations-find(tmp, 1, 'first')+1)./permutations;
                        nYprog=sum(~isnan(data{n}.y{4}));
                    else
                        pXprog(whichRoi, n)=0;
                        nXprog=sum(~isnan(data{n}.y{4}));
                    end
                    
                    ydat=log(data{n}.y{5}(~isnan(data{n}.y{5})));
                    xdat=data{n}.x{5}(~isnan(data{n}.y{5}));
                    fitDist=zeros(2,permutations);
                    for m=1:permutations
                        yshuffle=ydat(randperm(length(ydat)));
                        fitDist(:,m)=linreg(xdat, yshuffle);
                    end
                    CI=[prctile(fitDist(2,:), 2.5) prctile(fitDist(2,:), 97.5)];
                    max(fitDist(2,:));
                    measure=linreg(xdat, ydat);
                    measure(2);
                    [whichRoi, n]
                    if measure(2)<max(fitDist(2,:))
                        tmp= fitDist(2,:);
                        tmp=sort(tmp);
                        tmp=tmp>measure(2);
                        pYprog(whichRoi, n)=(permutations-find(tmp, 1, 'first')+1)./permutations;
                        nYprog=sum(~isnan(data{n}.y{5}));
                    else
                        pYprog(whichRoi, n)=0;
                        nYprog=sum(~isnan(data{n}.y{5}));
                    end
                end
            end
        end
        %Summary plot, linked points, no errorbars
        if showPlots(2)==1
            plotvec = [0.01:0.01:1];
            pct1 = 100*0.05/2;
            pct2 = 100-pct1;
               
            % Plot Duration
            try
            subplot(2,5,whichRoi)
            hold on;
            for n=1:length(data)
                if n==1
                    errorbar(data{n}.x{4}.*data{n}.meanDist,data{n}.y{4},data{n}.ysterr{4}(1,:),data{n}.ysterr{4}(1,:),strcat(colors{n},'o'),...
                        'MarkerFaceColor',colors{n},'MarkerEdgeColor','k','MarkerSize',8);
                    % Add Cyclical fit of Duration
                    extsd = bins(~isnan(data{1}.y{4}));
%                     plotsind = data{n}.logLineFitX(3).*sin(data{n}.logLineFitX(1).*plotvec+data{n}.logLineFitX(2))+polyval(linFitX, plotvec);
%                     plotsind(~(plotvec > extsd(1) & plotvec < extsd(end))) = NaN;
%                     hold on; plot(plotvec.*data{n}.meanDist, plotsind, colors{5},'LineWidth',2);
                    % Add Linear fit of Duration
                    linplotd = polyval(linFitX,bins);
                    linplotd(~(bins >= extsd(1) & bins <= extsd(end))) = NaN;
%                     hold on; plot(bins.*data{n}.meanDist, linplotd, colors{6}, 'LineWidth', 2);
                    hold on; plot(bins.*data{n}.meanDist, linplotd, '.-k', 'LineWidth', 2);
                else
                    hold on; plot(data{n}.x{4}.*data{n}.meanDist,data{n}.y{4}, colors{n},'LineWidth',2);
                end
            end
            axis([0 5*ceil(data{n}.meanDist/5) 0 1]);
            axis square;
            title([data{n}.ROItitle, ' Duration']);
            drawnow;
            catch
                fprintf('Plotting duration, 282-310.\n')
            end
%             % Start Period loop
%             try
%             subplot(4,5,whichRoi+10)
%             hold on;
%             for n=1:length(data)
%                 if n == 1
%                     errorbar(data{n}.x{5}.*data{n}.meanDist,data{n}.y{5},data{n}.ysterr{5}(1,:),data{n}.ysterr{5}(1,:),strcat(colors{n},'o'),...
%                         'MarkerFaceColor',colors{n},'MarkerEdgeColor','k','MarkerSize',8);
%                     extsp = bins(~isnan(data{1}.y{5}));
%                     
%                     % Add Cyclical fit of Duration
% %                     plotsinp = data{n}.logLineFitY(3).*sin(data{n}.logLineFitY(1).*plotvec+data{n}.logLineFitY(2))+polyval(linFitY, plotvec);
% %                     plotsinp(~(plotvec > extsp(1) & plotvec < extsp(end))) = NaN;
% %                     hold on; plot(plotvec.*data{n}.meanDist, plotsinp, colors{5},'LineWidth',2);
% 
%                     % Add Linear fit of Period
%                     linplotp = polyval(linFitY,bins);
%                     linplotp(~(bins >= extsp(1) & bins <= extsp(end))) = NaN;
%                     hold on; plot(bins.*data{n}.meanDist, linplotp, colors{6}, 'LineWidth', 2);
%                     ylim = [0, 2];     
%                 else
%                     hold on; plot(data{n}.x{5}.*data{n}.meanDist,data{n}.y{5}, colors{n},'LineWidth',2);
%                 end
%             end
%             axis([0 5*ceil(data{n}.meanDist/5) 0 1.2]);
%             axis square;
%             title([data{n}.ROItitle, ' Period']);
%             drawnow;
% %             saveas(gcf, ['PeriodDistance' data{n}.ROItitle, 'No0Hz'], 'epsc');
%             catch
%                 fprintf('Plotting Period, 314-338.\n')
%             end
        end
        
        
        if showPlots(3)==1;
            try
            figure; plot(data{1}.x0s(data{1}.veIndices), data{1}.y0s(data{1}.veIndices), 'k.');
            line=linreg(data{1}.x0s(data{1}.veIndices), data{1}.y0s(data{1}.veIndices))
            hold on; plot([0 1], [line(1) line(1)+line(2)], 'k');
            axis([0 1 0 1])
            axis square;
            title(['PeriodDuration' data{n}.ROItitle]);
            saveas(gcf, ['PeriodDuration' data{n}.ROItitle], 'epsc');
            catch
                fprintf('ShowPlots 3')
            end
        end
        
        % Correlations and pVals for x-Validation
        try
        indices=data{2}.veIndices & data{3}.veIndices;
        if sum(indices)<2
           rDurOddEven(whichRoi) = 0;
           rPerOddEven(whichRoi) = 0;
        else
        tmp=corrcoef(data{2}.x0s(indices), data{3}.x0s(indices));
        rDurOddEven(whichRoi)=tmp(1,2);
        tmp=corrcoef(data{2}.y0s(indices), data{3}.y0s(indices));
        rPerOddEven(whichRoi)=tmp(1,2);
        end
        nsOddEven(whichRoi)=length(data{2}.x0s(indices))./1.77^2;
        pDurOddEven(whichRoi)=r2p(rDurOddEven(whichRoi), nsOddEven(whichRoi));
        pPerOddEven(whichRoi)=r2p(rPerOddEven(whichRoi), nsOddEven(whichRoi));
        catch
            fprintf('Odd-Even comparison 358-373')
        end
        
        
        if showPlots(4)==1;
            try
            figure; plot(data{2}.x0s(indices), data{3}.x0s(indices), 'k.');
            hold on; plot(data{2}.y0s(indices), data{3}.y0s(indices), 'b.');
            line=linreg(data{2}.x0s(indices), data{3}.x0s(indices))
            hold on; plot([0 1], [line(1) line(1)+line(2)], 'k');
            line=linreg(data{2}.y0s(indices), data{3}.y0s(indices))
            hold on; plot([0 1], [line(1) line(1)+line(2)], 'b');
            axis([0 1 0 1])
            axis square;
            title(['OddEvenCor' data{n}.ROItitle]);
            saveas(gcf, ['OddEvenCor' data{n}.ROItitle], 'epsc');
            catch
                fprintf('showPlots = 4, print at 348')
            end
        end
        
        if showPlots(5)==1;
            try
            [X,Y] = meshgrid(0:0.001:1,0:0.001:1);
            veIndices=find(data{1}.veIndices);
            rf=zeros([length(X(:)), length(veIndices)]) ;
            for whichRF=1:length(veIndices)
                rftmp   = rfGaussian2d(X(:), Y(:),...
                    data{1}.sigmas(veIndices(whichRF)), ...
                    data{1}.sigmaMinor(veIndices(whichRF)), ...
                    data{1}.sigmaTheta(veIndices(whichRF)), ...
                    data{1}.x0s(veIndices(whichRF)), ...
                    data{1}.y0s(veIndices(whichRF)));
                rf(:,whichRF)=rftmp;
            end
            
            vol = data{1}.sigmas(data{1}.veIndices).*data{1}.sigmaMinor(data{1}.veIndices).*data{1}.ves(data{1}.veIndices);
            vol = vol * (2 * pi);
            rf=rf./(ones(size(rf)) * vol');
            [~, rfPeaks]=max(rf, [], 1);
            rfall=max(rf, [], 2);
            rfall(rfPeaks)=min(rfall(:));
            rfall=reshape(rfall, size(X));
            figure; imagesc(flipud(rfall)); colormap hot; axis square
            saveas(gcf, ['Coverage' data{n}.ROItitle], 'epsc');
            catch
                fprintf('showPlots == 5, print at 377')
            end
        end
    end
end
saveas(gcf, ['DurationPeriodDistance' data{n}.ROItitle], 'epsc');
catch
   fprintf('Monster Loop ONE at 24-391.\n')    
end



try
if showPlots(1)==1
    stats.pXprog=pXprog;
    stats.pYprog=pYprog;
    stats.nXprog=nXprog;
    stats.nYprog=nYprog;  
end

stats.rDistDurCorr=rDistDurCorr;
stats.rDistPerCorr=rDistPerCorr;
stats.ns=ns;
stats.pDistDurCorr=pDistDurCorr;
stats.pDistPerCorr=pDistPerCorr;
stats.rDurPerCorr=rDurPerCorr;
stats.pDurPerCorr=pDurPerCorr;

stats.rDurOddEven=rDurOddEven;
stats.rPerOddEven=rPerOddEven;
stats.nsOddEven=nsOddEven;
stats.pDurOddEven=pDurOddEven;
stats.pPerOddEven=pPerOddEven;

%All left then all right
%colors={[0 0 255]./255, [128 0 255]./255, [255 0 255]./255, [255 0 0]./255, [255 128 0]./255, [128 64 0]./255, [128 128 128]./255, [0 128 0]./255, ...
%   [0 0 255]./255, [128 0 255]./255, [255 0 255]./255, [255 0 0]./255, [255 128 0]./255, [128 64 0]./255, [128 128 128]./255, [0 128 0]./255 };
%Alternating hemispheres
colors={[0 0 255]./255, [0 0 255]./255, [128 0 255]./255, [128 0 255]./255, [255 0 255]./255, [255 0 255]./255, [255 0 0]./255, [255 0 0]./255, [255 128 0]./255, [255 128 0]./255, [128 64 0]./255, [128 64 0]./255, [128 128 128]./255, [128 128 128]./255, [0 128 0]./255, [0 128 0]./255 , [0 255 0]./255, [0 255 0]./255};
%fwhm vs number
catch
    fprintf('Storing Stats at 396-425.\n')
end



try
for n=1:size(dataAll,1)        
    try
        if ~isempty(dataAll{n,1})
            [~, ~, T, P, df]=lineFitterV(1:sum(dataAll{n,1}.veIndices), dataAll{n,1}.x0s(dataAll{n,1}.veIndices),dataAll{n,1}.sigmas(dataAll{n,1}.veIndices),ones(size(dataAll{n,1}.sigmas(dataAll{n,1}.veIndices))), 1.77.^2);
            tTuningWidthMajor(n,:)=T;
            pTuningWidthMajor(n,:)=P;
            nTuningWidth(n)=df+3;
            nUnderOverTuningWidth(n,:)=[sum(dataAll{n,1}.x0s(dataAll{n,1}.veIndices)<0.5)./1.77^2 sum(dataAll{n,1}.x0s(dataAll{n,1}.veIndices)>0.5)./1.77^2];
            [~, ~, T, P, df]=lineFitterV(1:sum(dataAll{n,1}.veIndices), dataAll{n,1}.x0s(dataAll{n,1}.veIndices),dataAll{n,1}.sigmaMinor(dataAll{n,1}.veIndices),ones(size(dataAll{n,1}.sigmas(dataAll{n,1}.veIndices))), 1.77.^2);
            tTuningWidthMinor(n,:)=T;
            pTuningWidthMinor(n,:)=P;
            
            if showPlots(6)==1
                
                
                bins=linspace(0, 1.05, 22);
                dataAll{n,1}.x{8}=bins;
                dataAll{n,1}.y{8}=nan(size(bins));
                dataAll{n,1}.ysterr{8}=nan(2,length(bins));
                dataAll{n,1}.x{9}=bins;
                dataAll{n,1}.y{9}=nan(size(bins));
                dataAll{n,1}.ysterr{9}=nan(2,length(bins));
                %         dataAll{n,1}.yLog{9}=nan(size(b));
                %         dataAll{n,1}.ysterrLog{9}=nan(2,length(b));
                for bCount=1:length(bins)
                    % Populate bin indices based on duration criteria
                    bii = dataAll{n,1}.x0s> bins(bCount)- bins(2)/2 & ...
                        dataAll{n,1}.x0s < bins(bCount)+ bins(2)/2 & ...
                        dataAll{n,1}.sigmas<=30 & dataAll{n,1}.veIndices;
                    if any(bii) && sum(dataAll{n,1}.ves(bii))>0 && numel(dataAll{n,1}.sigmas(bii))>1.77^2;
                        % Populate duration bins with tuningWidth data
                        s=wstat(dataAll{n,1}.sigmas(bii), dataAll{n,1}.ves(bii), 1.77^2);
                        dataAll{n,1}.y{8}(bCount)=s.mean;
                        dataAll{n,1}.ysterr{8}(:,bCount)=s.sterr;
                        s=wstat(dataAll{n,1}.sigmaMinor(bii), dataAll{n,1}.ves(bii), 1.77^2);
                        dataAll{n,1}.y{9}(bCount)=s.mean;
                        dataAll{n,1}.ysterr{9}(:,bCount)=s.sterr;
                        
                    end
                end
%                 for bCount=1:length(bins)
%                     bii = data{n}.ratio> bins(bCount)-bins(1) & ...
%                         data{n}.ratio< bins(bCount)+bins(1) & data{n}.veIndices;
%                     if any(bii)
%                         s=wstat(data{n}.x0s(bii), data{n}.ves(bii), 1.77^2);
%                         data{n}.y{4}(bCount)=s.mean;
%                         data{n}.ysterr{4}(:,bCount)=s.sterr;
%                         s=wstat(data{n}.y0s(bii), data{n}.ves(bii), 1.77^2);
%                         data{n}.y{5}(bCount)=s.mean;
%                         data{n}.ysterr{5}(:,bCount)=s.sterr;
%                     end
%                 end              


            xpointsSigma=min(bins(~isnan(dataAll{n,1}.y{9}))):0.01:max(bins(~isnan(dataAll{n,1}.y{9})));
            dataAll{n,1}.xpointsSigma = xpointsSigma;                
                
                %data(n).fwhmnumfit=linreg(dataAll{n,1}.x{9}(isfinite(1./dataAll{n,1}.y{sterr9}(1,:))), dataAll{n,1}.y{9}(isfinite(1./dataAll{n,1}.y{sterr9}(1,:))), 1./dataAll{n,1}.y{sterr9}(1,isfinite(1./dataAll{n,1}.y{sterr9}(1,:))));
                [dataAll{n,1}.ypointsSigMajor, dataAll{n,1}.SigMajorNumFit, dataAll{n,1}.SigMajorbupper, dataAll{n,1}.SigMajorblower]=bootstrapLineFitterV(dataAll{n,1}.x{8},dataAll{n,1}.y{8},1./dataAll{n,1}.ysterr{8}(1,:), dataAll{n,1}.xpointsSigma);
                [dataAll{n,1}.ypointsSigMinor, dataAll{n,1}.SigMinorNumFit, dataAll{n,1}.SigMinorbupper, dataAll{n,1}.SigMinorblower]=bootstrapLineFitterV(dataAll{n,1}.x{9},dataAll{n,1}.y{9},1./dataAll{n,1}.ysterr{9}(1,:), dataAll{n,1}.xpointsSigma);
%                 [dataAll{n,1}.ypointsSigMajor, dataAll{n,1}.SigMajorNumFit, dataAll{n,1}.SigMajorbupper, dataAll{n,1}.SigMajorblower]=bootstrapLineFitter(dataAll{n,1}.x{8},dataAll{n,1}.y{8},1./dataAll{n,1}.ysterr{8}(1,:), dataAll{n,1}.xpointsSigma);
%                 [dataAll{n,1}.ypointsSigMinor, dataAll{n,1}.SigMinorNumFit, dataAll{n,1}.SigMinorbupper, dataAll{n,1}.SigMinorblower]=bootstrapLineFitter(dataAll{n,1}.x{9},dataAll{n,1}.y{9},1./dataAll{n,1}.ysterr{9}(1,:), dataAll{n,1}.xpointsSigma);
                
                if showPlots(7)==1
                    permutations=10000;
                    ydat=dataAll{n,1}.y{8}(~isnan(dataAll{n,1}.y{8}));
                    xdat=bins(~isnan(dataAll{n,1}.y{8}));
                    fitDist=zeros(3,permutations);
                    for m=1:permutations
                        yshuffle=ydat(randperm(length(ydat)));
                        fitDist(:,m)=lineFitterV(1:length(xdat), xdat,yshuffle,ones(size(xdat)));
                    end
                    CI=[prctile(fitDist(1:2,:)', 2.5); prctile(fitDist(1:2,:)', 97.5)];
                    measure=lineFitterV(1:length(xdat), xdat,ydat,ones(size(xdat)));
                    
                    if measure(1)<max(fitDist(1,:))
                        tmp= fitDist(1,:);
                        tmp=sort(tmp);
                        tmp=tmp>measure(1);
                        pSMajProg(n,1)=(permutations-find(tmp, 1, 'first')+1)./permutations;
                    else
                        pSMajProg(n,1)=0;
                    end
                    if measure(2)<max(fitDist(2,:))
                        tmp= fitDist(2,:);
                        tmp=sort(tmp);
                        tmp=tmp>measure(2);
                        pSMajProg(n,2)=(permutations-find(tmp, 1, 'first')+1)./permutations;
                    else
                        pSMajProg(n,2)=0;
                    end
                    
                    ydat=dataAll{n,1}.y{9}(~isnan(dataAll{n,1}.y{9}));
                    xdat=bins(~isnan(dataAll{n,1}.y{9}));
                    fitDist=zeros(3,permutations);
                    for m=1:permutations
                        yshuffle=ydat(randperm(length(ydat)));
                        fitDist(:,m)=lineFitterV(1:length(xdat), xdat,yshuffle,ones(size(xdat)));
                    end
                    CI=[prctile(fitDist(1:2,:)', 2.5); prctile(fitDist(1:2,:)', 97.5)];
                    measure=lineFitterV(1:length(xdat), xdat,ydat,ones(size(xdat)));
                    
                    if measure(1)<max(fitDist(1,:))
                        tmp= fitDist(1,:);
                        tmp=sort(tmp);
                        tmp=tmp>measure(1);
                        pSMinProg(n,1)=(permutations-find(tmp, 1, 'first')+1)./permutations;
                    else
                        pSMinProg(n,1)=0;
                    end
                    if measure(2)<max(fitDist(2,:))
                        tmp= fitDist(2,:);
                        tmp=sort(tmp);
                        tmp=tmp>measure(2);
                        pSMinProg(n,2)=(permutations-find(tmp, 1, 'first')+1)./permutations;
                    else
                        pSMinProg(n,2)=0;
                    end
                end
                
                
                %Linear fit only
                %         if showPlots(7)==1
                %             permutations=10000;
                %             ydat=dataAll{n,1}.y{8}(~isnan(dataAll{n,1}.y{8}));
                %             xdat=b(~isnan(dataAll{n,1}.y{8}));
                %             fitDist=zeros(2,permutations);
                %             for m=1:permutations
                %                 yshuffle=ydat(randperm(length(ydat)));
                %                 fitDist(:,m)=linreg(xdat, yshuffle);
                %             end
                %             CI=[prctile(fitDist(2,:), 2.5) prctile(fitDist(2,:), 97.5)];
                %             max(fitDist(2,:));
                %             measure=linreg(xdat, ydat);
                %             measure(2);
                %             if measure(2)<max(fitDist(2,:))
                %                 tmp= fitDist(2,:);
                %                 tmp=sort(tmp);
                %                 tmp=tmp>measure(2);
                %                 pSMajProg(n)=(permutations-find(tmp, 1, 'first')+1)./permutations;
                %             else
                %                 pSMajProg(n)=0;
                %             end
                %
                %             ydat=dataAll{n,1}.y{9}(~isnan(dataAll{n,1}.y{9}));
                %             xdat=b(~isnan(dataAll{n,1}.y{9}));
                %             fitDist=zeros(2,permutations);
                %             for m=1:permutations
                %                 yshuffle=ydat(randperm(length(ydat)));
                %                 fitDist(:,m)=linreg(xdat, yshuffle);
                %             end
                %             CI=[prctile(fitDist(2,:), 2.5) prctile(fitDist(2,:), 97.5)];
                %             max(fitDist(2,:));
                %             measure=linreg(xdat, ydat);
                %             measure(2);
                %             if measure(2)<max(fitDist(2,:))
                %                 tmp= fitDist(2,:);
                %                 tmp=sort(tmp);
                %                 tmp=tmp>measure(2);
                %                 pSMinProg(n)=(permutations-find(tmp, 1, 'first')+1)./permutations;
                %             else
                %                 pSMinProg(n)=0;
                %             end
                %         end
            end
        end
    catch
        n
    end


        
    %for n=1:size(dataAll,1)%:length(data)
    try
        if ~isempty(dataAll{n,1}) && showPlots(6)==1
            figure;
            pause(0.00001);
            frame_h = get(handle(gcf),'JavaFrame');
            set(frame_h,'Maximized',1);
            hold on;
            
%             plot(exp(data(n).x0s(data(n).veIndices)), fwhms(data(n).veIndices), strcat(colors{n}, 'o'), 'MarkerSize',3, 'MarkerFaceColor', colors{n})
            %fitIndices=data(n).veIndices & dataAll{n,1}.sigmas<30 & data(n).x0s<=7;
            
            hold on; plot(dataAll{n,1}.xpointsSigma, dataAll{n,1}.ypointsSigMajor, 'Color', colors{n},'LineWidth',3);
            hold on; plot(dataAll{n,1}.xpointsSigma, dataAll{n,1}.ypointsSigMinor, ':', 'Color', colors{n},'LineWidth',2);
            
            
            errorbar(bins,dataAll{n,1}.y{8},dataAll{n,1}.ysterr{8}(1,:),dataAll{n,1}.ysterr{8}(1,:),strcat('o'),...
                'Color', colors{n},'MarkerFaceColor',colors{n},'MarkerEdgeColor','k','MarkerSize',8);
            
            errorbar(bins,dataAll{n,1}.y{9},dataAll{n,1}.ysterr{9}(1,:),dataAll{n,1}.ysterr{9}(1,:),strcat('o'),...
                'Color', colors{n},'MarkerFaceColor',colors{n},'MarkerEdgeColor','k','MarkerSize',6);
            
            
            hold on; plot(dataAll{n,1}.xpointsSigma, dataAll{n,1}.SigMajorbupper, 'Color', colors{n});
            hold on; plot(dataAll{n,1}.xpointsSigma, dataAll{n,1}.SigMajorblower, 'Color', colors{n});
            
            hold on; plot(dataAll{n,1}.xpointsSigma, dataAll{n,1}.SigMinorbupper, 'Color', colors{n});
            hold on; plot(dataAll{n,1}.xpointsSigma, dataAll{n,1}.SigMinorblower, 'Color', colors{n});
            
            %plot(exp(data(n).x0s(data(n).veIndices)), fwhms(data(n).veIndices), strcat(colors{n}, 'o'), 'MarkerSize',3, 'MarkerFaceColor', colors{n})
            %fitIndices=data(n).veIndices & dataAll{n,1}.sigmas<30 & data(n).x0s<=7;
            
            
            
            %hold on; plot([0 7], [data(n).fwhmnumfit(1), data(n).fwhmnumfit(1)+data(n).fwhmnumfit(2)*7], colors{n});
            %     xpoints=0:0.01:1;
            %     ypoints=x0fit(1)+xpoints.*x0fit(2);
            %     hold on; plot(xpoints, exp(ypoints));
            title(dataAll{n,1}.ROItitle)
            axis([0 1 0 2]);
            legend('SigmaMajor', 'SigmaMinor')
            axis square;
            drawnow;%
            saveas(gcf, ['TuningWidthDuration' dataAll{n,1}.ROItitle], 'epsc');
%             saveas(gcf, ['TuningWidthDuration' dataAll{n,1}.ROItitle], 'epsc');
        end
    catch
        n
    end

        %saveas(gcf, ['TuningWidthDuration' dataAll{n,1}.ROItitle], 'epsc');
        
        %Summary plot
%         figure;
%         hold on;
%         for n=1:size(dataAll,1)
%             dataAll{n,1}.x{9}=b(isfinite(dataAll{n,1}.y{9}));
%             plot(dataAll{n,1}.x{9},dataAll{n,1}.y{9}(isfinite(dataAll{n,1}.y{9})), 'Color', colors{n},'LineWidth',2);
%         end
%         axis([0 6 0 20]);
%         axis square
end
catch
   fprintf('During SECOND uncommented monster loop loop')    
end

stats.tTuningWidthMajor=tTuningWidthMajor;
stats.pTuningWidthMajor=pTuningWidthMajor;
stats.nTuningWidth=nTuningWidth;
stats.nUnderOverTuningWidth=nUnderOverTuningWidth;
stats.tTuningWidthMinor=tTuningWidthMinor;
stats.pTuningWidthMinor=pTuningWidthMinor;
    
if showPlots(7)==1
    stats.pSMajProg=pSMajProg;
    stats.pSMinProg=pSMinProg;
    
    for n=1:18;
        try
            tuningSlopesMajor(n,:)=dataAll{n,1}.SigMajorNumFit;
            tuningSlopesMinor(n,:)=dataAll{n,1}.SigMinorNumFit;
            tuningXs(n,:)=dataAll{n,1}.x{8};
            tuningYmajor(n,:)=dataAll{n,1}.y{8};
            tuningYminor(n,:)=dataAll{n,1}.y{9};
        catch
            fprintf('647-654\n')
        end
    end
    stats.tuningSlopesMajor=tuningSlopesMajor;
    stats.tuningSlopesMinor=tuningSlopesMinor;
    stats.tuningXs=tuningXs;
    stats.tuningYmajor=tuningYmajor;
    stats.tuningYminor=tuningYminor;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ypoints, yfitParams, b_upper, b_lower,B]=bootstrapLogLineFitter(x,y,ve, xpoints)
x = x(isfinite(ve)); y = y(isfinite(ve)); ve = ve(isfinite(ve));
%ve=ones(size(ve));
iis=1:length(x);
[B] = bootstrp(1000,@(iis) logLineFitter(iis,x,y,ve),[1:length(x)]');
B = B';
roi.p=B;
pct1 = 100*0.05/2;
pct2 = 100-pct1;
b_lower = prctile(B',pct1);
b_upper = prctile(B',pct2);
yfitParams=prctile(B', 50);
%y2fit = polyval(roi.p2(:),x2fit);
keep1 = B(1,:)>b_lower(1) &  B(1,:)<b_upper(1);
keep2 = B(2,:)>b_lower(2) &  B(2,:)<b_upper(2);
keep = keep1 & keep2;

% ypoints=exp(yfitParams(2)+xpoints.*yfitParams(1));
% ypoints=yfitParams(5).*sin(yfitParams(1).*xpoints+yfitParams(2))+yfitParams(3).*xpoints+yfitParams(4);
% ypoints=(sin(yfitParams(1).*xpoints+yfitParams(2))+yfitParams(3));
ypoints=yfitParams(3).*(sin(yfitParams(1).*xpoints+yfitParams(2))); 
% A == yfitparams(3), scaling
% f == yfitparams(1), frequency
% phi == yfitparams(2), phase


% ypoints=real(yfitParams(4).*(exp(1i*(yfitParams(1).*xpoints+yfitParams(2))+yfitParams(3))));
% fits = exp([xpoints' ones(size(xpoints'))]*B(:,keep));
% fits = [ones(size(xpoints'))]*B(5,keep).*sin([xpoints' ones(size(xpoints'))]*B([1,2],keep))+[xpoints' ones(size(xpoints'))]*B([3,4],keep);
% fits = [ones(size(xpoints'))*B(4,keep)].*(sin(([xpoints' ones(size(xpoints'))]*B([1,2],keep))+ones(size(xpoints'))*B(3,keep)));
fits = [ones(size(xpoints'))*B(3,keep)].*(sin(xpoints'*B(1,keep)+ones(size(xpoints'))*B(2,keep)));
% fits = real([ones(size(xpoints'))*B(4,keep)].*(exp(1i*([xpoints' ones(size(xpoints'))]*B([1,2],keep))+ones(size(xpoints'))*B(3,keep))));
b_upper = max(fits,[],2);
b_lower = min(fits,[],2);
end

function [B,e]=logLineFitter(iis, x,y,ve)
%x = x(isfinite(ve)); y = y(isfinite(ve)); ve = ve(isfinite(ve));
x=x(iis);
y=y(iis);
ve=ve(iis);
options = optimset('Display','off','MaxFunEvals',10000000, 'InitTrustRegionRadius', 10, 'Algorithm', 'interior-point-convex');%,  'FunValCheck', 'on', 'Diagnostics', 'on');%'UseParallel', true,
% [B,e] = fminsearch(@(z) mylogfit(z,x,y,ve),[0.2;1.3], options); % for logline
% [B,e] = fminsearch(@(z) mylogfit(z,x,y,ve),[10;0;0.2;1.3;0.01], options); % for sine + linear
[B,e] = fminsearch(@(z) mylogfit(z,x,y,ve),[25;-2;0.15], options);

end

function e=mylogfit(z,x,y,ve)
e=sum(ve.*(y-(z(3).*(sin(z(1).*x+z(2))))).^2)./sum(ve);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ypoints, yfitParams, b_upper, b_lower, B]=bootstrapLineFitter(x,y,ve, xpoints)
x = x(isfinite(ve)); y = y(isfinite(ve)); ve = ve(isfinite(ve));
ve=ones(size(ve));
iis=1:length(x);
[B] = bootstrp(1000,@(iis) lineFitter(iis,x,y,ve),[1:length(x)]');
B = B';
roi.p=B;
pct1 = 100*0.05/2;
pct2 = 100-pct1;
% Parameters interact, rewrite at some point
b_lower = prctile(B',pct1);
b_upper = prctile(B',pct2);
yfitParams=prctile(B', 50);
%y2fit = polyval(roi.p2(:),x2fit);
keep1 = B(1,:)>b_lower(1) &  B(1,:)<b_upper(1);
keep2 = B(2,:)>b_lower(2) &  B(2,:)<b_upper(2);
keep = keep1 & keep2;

ypoints=yfitParams(2)+xpoints.*yfitParams(1);
fits = [xpoints' ones(size(xpoints'))]*B(:,keep);
b_upper = max(fits,[],2);
b_lower = min(fits,[],2);
end

function [B,e]=lineFitter(iis, x,y,ve)
%x = x(isfinite(ve)); y = y(isfinite(ve)); ve = ve(isfinite(ve));
x=x(iis);
y=y(iis);
ve=ve(iis);
options = optimset('MaxFunEvals',10000000);
[B,e] = fminsearch(@(z) mylinfit(z,x,y,ve),[0.2;1.3], options);
end

function e=mylinfit(z,x,y,ve)
e=sum(ve.*(y-((z(1).*x+z(2)))).^2)./sum(ve);
%e=sum(ve.*(y-(1./(z(1).*x+z(2)))).^2)./sum(ve);
end


function [ypoints, yfitParams, b_upper, b_lower, B]=bootstrapLineFitterV(x,y,ve, xpoints)
x = x(isfinite(ve)); y = y(isfinite(ve)); ve = ve(isfinite(ve));
ve=ones(size(ve));
iis=1:length(x);
[B] = bootstrp(1000,@(iis) lineFitterV(iis,x,y,ve),[1:length(x)]');
B = B';
roi.p=B;
pct1 = 100*0.05/2;
pct2 = 100-pct1;
b_lower = prctile(B',pct1);
b_upper = prctile(B',pct2);
yfitParams=prctile(B', 50);
%y2fit = polyval(roi.p2(:),x2fit);
keep1 = B(1,:)>b_lower(1) &  B(1,:)<b_upper(1);
keep2 = B(2,:)>b_lower(2) &  B(2,:)<b_upper(2);
keep3 = B(3,:)>b_lower(3) &  B(3,:)<b_upper(3);
keep = keep1 & keep2 & keep3;

ypoints=yfitParams(3)+xpoints.*yfitParams(1)+abs(0.5-xpoints).*yfitParams(2);
fits = [xpoints' abs(0.5-xpoints') ones(size(xpoints'))]*B(:,keep);
b_upper = max(fits,[],2);
b_lower = min(fits,[],2);
end

function [B, e, T, P, df]=lineFitterV(iis, x,y,ve, upsample)
%x = x(isfinite(ve)); y = y(isfinite(ve)); ve = ve(isfinite(ve));
if ~exist('upsample', 'var') || isempty(upsample)
    upsample=1;
end
x=x(iis);
y=y(iis);
ve=ve(iis);
designMatrix=[x' abs(0.5-x)' ones(size(x))'];
B=pinv(designMatrix)*y';
pred=designMatrix*B;
pred=pred';

df=floor(length(iis)./upsample)-size(designMatrix, 2);
T=zeros(1, size(designMatrix,2));
for n=1:size(designMatrix, 2)
    c=zeros(1, size(designMatrix,2));
    c(n)=1;
    
    SE=sqrt((sum((y-pred).^2)./df)*(c*pinv(designMatrix'*designMatrix)*c'));
    T(n)=c*B./SE;
    P(n) = 2*tpvalue(-abs(T(n)),df);
end

e=sum(ve.*(y-pred).^2)./sum(ve);
end


function [ypoints, yfitParams, b_upper, b_lower] = bootstrapCmfFitter(x, y, ve, xpoints)
x = x(isfinite(ve)); y = y(isfinite(ve)); ve = ve(isfinite(ve));
iis=1:length(x);
[B] = bootstrp(1000,@(iis) cmfLineFitter(iis,x,y,ve),[1:length(x)]');
B = B';
roi.p=B;
pct1 = 100*0.05/2;
pct2 = 100-pct1;
b_lower = prctile(B',pct1);
b_upper = prctile(B',pct2);
yfitParams=prctile(B', 50);
%y2fit = polyval(roi.p2(:),x2fit);
keep1 = B(1,:)>b_lower(1) &  B(1,:)<b_upper(1);
keep2 = B(2,:)>b_lower(2) &  B(2,:)<b_upper(2);
keep = keep1 & keep2;

ypoints=1./(xpoints.*yfitParams(1)+yfitParams(2));
fits = 1./([xpoints' ones(size(xpoints'))]*B(:,keep));
b_upper = max(fits,[],2);
b_lower = min(fits,[],2);
end


function [B, e]=cmfLineFitter(ii,x,y,ve)
x = x(ii); y = y(ii); ve = ve(ii);
options = optimset('MaxFunEvals',10000000);
[B, e] = fminsearch(@(z) mycmffit(z,x,y,ve),[0.05;0.2], options);
end

function e=mycmffit(z,x,y,ve)
e=sum(ve.*(y-(1./(z(1).*x+z(2)))).^2)./sum(ve);
%phi, omega == z(1)&, z(2); intecept AND slope
% e=sum(ve.*(equation).^2)./sum(ve);
% e=sum(ve.*(sin(z(1)+z(2))).^2)./sum(ve);

end
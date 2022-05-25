%Plots bar chart and pairwise differences between candidate model fits.
clear barData targetDataOdd targetDataEven

bfp = "/mnt/data/matlab/bensfunctions_complete/";
vsp = "/mnt/data/vistasoftOld";
sp = "/mnt/data/";
sp2 = "/mnt/data/Try_full_run/";

addpath(genpath(bfp));
addpath(genpath(vsp));
addpath(sp);
addpath(sp2);


loadallnames;
mapNames = ["HeschlsGyrus","AnteriorBelt","PosteriorBelt","MiddleBelt","Premotor"];
models = {'Lin-2dOvalGaussian-DurationPeriod-DT0.5-maxValue-2-expIntensity-free',
%     'LinMonotonic-DurationPeriod-DT0.5',
%     'LinNormLogMonotonic-DurationPeriodDT0.5',
    'Lin-1dGaussianXlinearY-DurFreq-20',
    'Lin-linearXcompressiveYNoNormOccupancy-DurFreq-20',
    'Lin-linearXlinearYNoNormOccupancy-DurFreq-20',
    'Lin-1dGaussianXcompressiveY-OccupancyFreq-20',
    'Lin-compressiveXcompressiveYNoNormOccupancy-DurFreq-20',
    'Lin-Flat-FreqOnly-20',
    'Log-2dCircGaussian-OccupancyPeriod-DT0.5-maxValue-4.8-logIntensity-1',
    'Log-2dOvalGaussian-OccupancyPeriod-DT0.5-maxValue-3.8-logIntensity-1',
    'Lin-2dOvalGaussian-OccupancyPeriod-DT0.5-maxValue-2-expIntensity-1',
    'Lin-2dOvalGaussian-OccupancyPeriod-DT0.5-maxValue-2-expIntensity-free',
    'LinNormLog-2dOvalGaussian-OccupancyPeriod-DT0.5-maxValue-4-logIntensity-1',
    'Lin-2dCircGaussian-OccupancyPeriod-DT0.5-maxValue-2-logIntensity-1',
    'Log-2dOvalGaussian-DurationPeriod-DT0.5-maxValue-3.8-expIntensity-free'
    };

% mapNames = ["All"];
subjectOrder = DurationSubjectNames;

modelNamesAll = models;

% modelNamesAll =     {'Lin2dOvalGaussianDurationPeriodexpIntensityfree',
%     'Lin1dGaussianXlinearYDurFreq20',
%     'LinlinearXcompressiveYNoNormOccupancyDurFreq20'      ,
%     'LinlinearXlinearYNoNormOccupancyDurFreq20'           ,
%     'Lin1dGaussianXcompressiveYOccupancyFreq20'           ,
%     'LincompressiveXcompressiveYNoNormOccupancyDurFreq20' ,
%     'LinFlatFreqOnly20'                                   ,
%     'Log2dCircGaussianOccupancyPeriodlogIntensity1'       ,
%     'Log2dOvalGaussianOccupancyPeriodlogIntensity1'       ,
%     'Lin2dOvalGaussianOccupancyPeriodexpIntensity1'       ,
%     'Lin2dOvalGaussianOccupancyPeriodexpIntensityfree'    ,
%     %     'LinNormLog2dOvalGaussianOccupancyPeriodlogIntensity1',
%     'Lin2dCircGaussianOccupancyPeriodlogIntensity1'       ,
% %     'Log2dOvalGaussianDurationPeriodexpIntensityfree'     ,
%     %     'LinMonotonicDurationPeriod'                          ,
%     'Lin1dGaussianXcompressiveYDurFreq20'                 ,
%     'Lin2dCircGaussianDurationPeriodexpIntensity1'     ,
%     'Lin2dCircGaussianDurationPeriodexpIntensityfree',%};
%     'Tonotopy'}


modelNamesAll =     {'Lin2dOvalGaussianDurationPeriodexpIntensityfree',
%     'Lin1dGaussianXlinearYDurFreq20',
    'LinlinearXcompressiveYNoNormOccupancyDurFreq20'      ,
    'LinlinearXlinearYNoNormOccupancyDurFreq20'           ,
%     'Lin1dGaussianXcompressiveYOccupancyFreq20'           ,
    'LincompressiveXcompressiveYNoNormOccupancyDurFreq20' ,
    'LinFlatFreqOnly20'                                   ,
    'Log2dCircGaussianOccupancyPeriodlogIntensity1'       ,
    'Log2dOvalGaussianOccupancyPeriodlogIntensity1'       ,
    'Lin2dOvalGaussianOccupancyPeriodexpIntensity1'       ,
    'Lin2dOvalGaussianOccupancyPeriodexpIntensityfree'    ,
%         'LinNormLog2dOvalGaussianOccupancyPeriodlogIntensity1',
%     'Lin2dCircGaussianOccupancyPeriodlogIntensity1'       ,
    'Log2dOvalGaussianDurationPeriodexpIntensityfree'     ,
    %     'LinMonotonicDurationPeriod'                          ,
    'Lin1dGaussianXcompressiveYDurFreq20'                 ,
    'Lin2dCircGaussianDurationPeriodexpIntensity1'     ,
    'Lin2dCircGaussianDurationPeriodexpIntensityfree'};
%     'Tonotopy'}



%%
thr= 0.1;

%
subjectOrder = DurationSubjectNames;
% subjectOrder = subjectOrder(1);
numROIs = [1:5];
% mapNames = mapNames(5);
m = 1:(length(modelNamesAll)); % whichModel
clear barDataMeans barPoints barMeans barStd barSerr CI95 barData targetDataOdd targetDataEven indices tMatrix
% modelNamesAll
%Determine where to find data in structure
for whichSub= 1:length(subjectOrder)
    for whichMap=numROIs%length(mapNames)
        for whichHemi=1:2
            for modelN=m%1:length(modelNamesAll)
                % for variance explained
                targetDataOdd{whichSub, whichMap, whichHemi, modelN}=char(strcat(subjectOrder{whichSub}, '.', mapNames{whichMap}, '.',hemispheres{whichHemi},'.', modelNamesAll{modelN}, '.Odd.vesXval'));
                targetDataEven{whichSub, whichMap,whichHemi, modelN}=char(strcat(subjectOrder{whichSub}, '.', mapNames{whichMap}, '.',hemispheres{whichHemi},'.', modelNamesAll{modelN}, '.Even.vesXval'));
%                 targetDataEven{whichSub, whichMap,whichHemi, modelN}=char(strcat(subjectOrder{whichSub}, '.', mapNames{whichMap}, '.',hemispheres{whichHemi},'.', modelNamesAll{modelN}, '.All.ves'));
            end
        end
    end
end

%Get data from these locations (cross validated variance explained)
barDataMeans=nan([length(subjectOrder), length(mapNames), 2,2,length(modelNamesAll)]);
% barDataMeans=nan([3, length(mapNames), 2,2,length(modelNamesAll)]);
for whichSub= 1:length(subjectOrder)
    for whichMap=numROIs%length(mapNames)
        if isfield(eval(subjectOrder{whichSub}), char(mapNames{whichMap}))
            for whichHemi=1:2;
                if isfield(eval([char(subjectOrder{whichSub}), '.', char(mapNames{whichMap})]), char(hemispheres{whichHemi}))
                    for modelN=m%1:length(modelNamesAll);
                        barData{whichSub, whichMap, whichHemi, 1}(:,modelN)=eval(targetDataOdd{whichSub, whichMap, whichHemi, modelN});
                        barData{whichSub, whichMap, whichHemi, 2}(:,modelN)=eval(targetDataEven{whichSub, whichMap, whichHemi, modelN});
                    end
                    for oddEven=1:2
                        indices=max(barData{whichSub, whichMap, whichHemi, oddEven},[],2)>=0.2;
                        barDataMeans(whichSub, whichMap, whichHemi, oddEven,:)=median(barData{whichSub, whichMap, whichHemi, oddEven}(indices,:), 1);
                    end
                end
            end
        end
    end
end

%Compute and plot means and confidence intervals of model fits
barPoints=[];
% whichBars=[1:length(modelNamesAll)];
% whichBars=[2,4,3,6,7,5,13,8,12,10,9,11,1];% sort by varexp
% whichBars = [3, 2, 10, 4, 11, 12, 5, 6, 8, 7, 9, 1];
whichBars = [1:length(modelNamesAll)];
% shortNames = {
%     "Dur./Per. Linear Oval Gaussian, FreeExp",
%     "Dur. Linear Gaussian & Monotonic Frequency",
%     "Monotonic Occ. & Compressive Frequency",
%     "Monotonic Occ. & Monotonic Frequency",
%     "Occ. Linear Gaussian & Compressive Frequency",
%     "Compressive Occ. & Compressive Frequency",
%     "Constant*Frequency",
%     "Occ./Per. Log Circular Gaussian",
%     "Occ./Per. Log Oval Gaussian",
%     "Occ./Per. Linear Oval Gaussian",
%     "Occ./Per. Linear Oval Gaussian, FreeExp",
%     "Norm Occ./Per. Log Oval Gaussian",
%     "Occ./Per. Linear Circular Gaussian",
%     "Dur./Per. Log Oval Gaussian, FreeExp",
%     "Monotonic Dur. & Monotonic Per."};

% NewBars = [7,4,3,6,15,2,5,13,8,10,9,12,11,1,14,16,17]; %Sorted by model complexity
% NewBars = [7,4,3,6,14,2,5,13,8,10,9,12,11,1,15,16]; %Sorted by model complexity
% tmpstor = barDataMeans(:,:,:,:,NewBars);

% NewBars = [3,7,2,6,1,5,15,4,16,8,10,7,12,9,11,14,13];
% whichBars = [6,3,5,2,14,4,15,16,12,7,8,11,9,10,13,1];

whichBars = [5, 3, 2, 4, 11, 12, 6, 13, 8, 7, 9, 10, 1];


modelNamesSort = modelNamesAll(whichBars);
% whichBars = NewBars;
% modelNamesSort = modelNamesAll(NewBars);
% modelNamesSort = shortNames(NewBars);

for n=m%1:length(whichBars)
    for nr = numROIs
    %     if n == 15
    %         continue;
    %     end
    tmp=barDataMeans(:, nr, :, :,n);
    %     tmp=barDataMeans(:, :, :, :,NewBars(n));
    %     tmp = tmpstor(:,:,:,:,n);
    tmp=(tmp(:));
    barPoints(:,n,nr)=tmp;
    tmp=tmp(~isnan(tmp));
    barMeans(n,nr)=mean(tmp);
    barStd(n,nr)=std(tmp);
    barSerr(n,nr)=std(tmp)/sqrt(length(tmp));
    CI95(n,:,nr) = tinv([0.025 0.975], length(tmp)-1);
    CI95(n,:,nr) =bsxfun(@times, barSerr(n,nr), CI95(n,:,nr));
    end
end
figure; 
% sortBars =[7,4,3,6,2,5,14,15,16,12,8,9,10,11,13,1];
% modelNamesSort = modelNamesAll;
% sortBars = 1:length(modelNamesAll);
sortBars = whichBars;
% [val, sortBars] = sort(barMeans);
for nr = numROIs
subplot(length(numROIs),2,1+2*(nr-1));
% tmp = barMeans(:,nr);
bar(barMeans(sortBars,nr));
% bar(barMeans([5,1:4]));
hold on; errorbar(1:size(barMeans(sortBars,nr),1), barMeans(sortBars,nr), CI95(sortBars,1,nr), CI95(sortBars,2,nr), 'k.')
axis square;
xlim([0.5 length(barMeans)+0.5])
ylim([0 0.4])
ax=gca;
ax.XTickLabelRotation=90;
ax.XTick = [1:length(modelNamesAll)];
set(ax,'xticklabel',modelNamesAll(sortBars));
ylabel('Median variance explained')

%Compute t-statistics and corresponding probabilities of pairwise
%differences between model fits\
for x=1:length(barMeans)
    %     for x=1:length(sortBars)
    for y=1:length(barMeans)
        [~,p,ci,stats] = ttest(barPoints(:,sortBars(x),nr), barPoints(:,sortBars(y),nr), 'tail', 'both');
        pvals(x,y,nr)=p;
        tMatrix(x,y,nr)=stats.tstat;
        %         tMatrix(x,y)=p;
    end
end
tMatrix(tMatrix>200)=200;
tMatrix(tMatrix<-200)=-200;

%This matrix uses 10*10 pixel cells, because some viewing software
%interpolates pixel edges
tMatrixImg=zeros(size(tMatrix,1)*10, size(tMatrix,2)*10);
for x=1:size(tMatrix,1)
    for y=1:size(tMatrix,1)
        tMatrixImg((x-1)*10+1:x*10, (y-1)*10+1:y*10)=tMatrix(x,y,nr);
    end
end
tMatrixImg(isnan(tMatrixImg))=0;
%Add image of resulting t-statistics
%figure;
subplot(length(numROIs),2,2*(nr));
% imagesc(tMatrixImg(end-59:end,end-59:end));

imagesc(tMatrixImg);
% imagesc(tMatrixImg);
clim = [-10 10];
ax=gca;
ax.XTickLabelRotation=90;
ax.XTick = [10:10:length(modelNamesAll)*10];
% set(ax,'xticklabel',modelNamesSort(end-5:end))
% set(ax,'xticklabel',modelNamesSort(sortBars))
ax.YTick = [10:10:length(modelNamesAll)*10];
% set(ax,'yticklabel',modelNamesSort(end-5:end))
% set(ax,'yticklabel',modelNamesSort(sortBars))
colormap(coolhotCmap([0], [128]));
colorbar;
axis image


end
saveas(gcf, ['ModelComparisonWithinROI'], 'epsc');
% saveas(gcf, ['ModelComparisonAllROIs'], 'epsc');


% Quickly remake indices
% make plot per ROI for duration and period for ONE EXAMPLE PARTICIPANT ONE
% HEMISPHERE
% vesindices =;
% figure;
% scatter(dataS1.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Even.x0s,dataS1.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Odd.x0s, '.')

% 
% 
% for n=m%1:length(whichBars)
%     for nr = 1:5
%     %     if n == 15
%     %         continue;
%     %     end
%     tmp=barDataMeans(:, nr, :, :,whichBars(n));
%     %     tmp=barDataMeans(:, :, :, :,NewBars(n));
%     %     tmp = tmpstor(:,:,:,:,n);
%     tmp=(tmp(:));
%     barPoints(:,nr)=tmp;
%     tmp=tmp(~isnan(tmp));
%     barMeans(nr)=mean(tmp);
%     barStd(nr)=std(tmp);
%     barSerr(nr)=std(tmp)/sqrt(length(tmp));
%     %     barPoints(:,n)=tmp;
%     CI95(nr,:) = tinv([0.025 0.975], length(tmp)-1);
%     CI95(nr,:) =bsxfun(@times, barSerr(nr), CI95(nr,:));
%     end
% end

%%
%%
thr= 0.1;

%
subjectOrder = DurationSubjectNames;
% subjectOrder = subjectOrder(1);
numROIs = [1:5];
% mapNames = mapNames(5);
m = 1:(length(modelNamesAll)); % whichModel
clear barDataMeans barPoints barMeans barStd barSerr CI95 barData targetDataOdd targetDataEven indices tMatrix
% modelNamesAll
%Determine where to find data in structure
for whichSub= 1:length(subjectOrder)
    for whichMap=numROIs%length(mapNames)
        for whichHemi=1:2
            for modelN=m%1:length(modelNamesAll)
                % for variance explained
                targetDataOdd{whichSub, whichMap, whichHemi, modelN}=char(strcat(subjectOrder{whichSub}, '.', mapNames{whichMap}, '.',hemispheres{whichHemi},'.', modelNamesAll{modelN}, '.Odd.vesXval'));
                targetDataEven{whichSub, whichMap,whichHemi, modelN}=char(strcat(subjectOrder{whichSub}, '.', mapNames{whichMap}, '.',hemispheres{whichHemi},'.', modelNamesAll{modelN}, '.Even.vesXval'));
%                 targetDataEven{whichSub, whichMap,whichHemi, modelN}=char(strcat(subjectOrder{whichSub}, '.', mapNames{whichMap}, '.',hemispheres{whichHemi},'.', modelNamesAll{modelN}, '.All.ves'));
            end
        end
    end
end

%Get data from these locations (cross validated variance explained)
barDataMeans=nan([length(subjectOrder), length(mapNames), 2,2,length(modelNamesAll)]);
% barDataMeans=nan([3, length(mapNames), 2,2,length(modelNamesAll)]);
for whichSub= 1:length(subjectOrder)
    for whichMap=numROIs%length(mapNames)
        if isfield(eval(subjectOrder{whichSub}), char(mapNames{whichMap}))
            for whichHemi=1:2;
                if isfield(eval([char(subjectOrder{whichSub}), '.', char(mapNames{whichMap})]), char(hemispheres{whichHemi}))
                    for modelN=m%1:length(modelNamesAll);
                        barData{whichSub, whichMap, whichHemi, 1}(:,modelN)=eval(targetDataOdd{whichSub, whichMap, whichHemi, modelN});
                        barData{whichSub, whichMap, whichHemi, 2}(:,modelN)=eval(targetDataEven{whichSub, whichMap, whichHemi, modelN});
                    end
                    for oddEven=1:2
                        indices=max(barData{whichSub, whichMap, whichHemi, oddEven},[],2)>=0.2;
                        barDataMeans(whichSub, whichMap, whichHemi, oddEven,:)=median(barData{whichSub, whichMap, whichHemi, oddEven}(indices,:), 1);
                    end
                end
            end
        end
    end
end

%Compute and plot means and confidence intervals of model fits
barPoints=[];
whichBars = [1:length(modelNamesAll)];
whichBars = [5, 3, 2, 4, 11, 12, 6, 13, 8, 7, 9, 10, 1];
modelNamesSort = modelNamesAll(whichBars);

for n=m%1:length(whichBars)
    for nr = numROIs
    tmp=barDataMeans(:, :, :, :,n);
    tmp=(tmp(:));
    barPoints(:,n)=tmp;
    tmp=tmp(~isnan(tmp));
    barMeans(n)=mean(tmp);
    barStd(n)=std(tmp);
    barSerr(n)=std(tmp)/sqrt(length(tmp));
    CI95(n,:) = tinv([0.025 0.975], length(tmp)-1);
    CI95(n,:) =bsxfun(@times, barSerr(n), CI95(n,:));
    end
end
figure; 
subplot(1,2,1)
% sortBars =[7,4,3,6,2,5,14,15,16,12,8,9,10,11,13,1];
% modelNamesSort = modelNamesAll;
% sortBars = 1:length(modelNamesAll);
sortBars = whichBars;
% [val, sortBars] = sort(barMeans);
% for nr = numROIs
% subplot(length(numROIs),2,1+2*(nr-1));
% tmp = barMeans(:,nr);
bar(barMeans(sortBars));
% bar(barMeans([5,1:4]));
hold on; errorbar(1:size(barMeans,2), barMeans(sortBars), CI95(sortBars,1), CI95(sortBars,2), 'k.')
axis square;
xlim([0.5 length(barMeans)+0.5])
ylim([0 0.3])
ax=gca;
ax.XTickLabelRotation=90;
ax.XTick = [1:length(modelNamesAll)];
set(ax,'xticklabel',modelNamesAll(sortBars));
ylabel('Median variance explained')

%Compute t-statistics and corresponding probabilities of pairwise
%differences between model fits\
for x=1:length(barMeans)
    %     for x=1:length(sortBars)
    for y=1:length(barMeans)
        [~,p,ci,stats] = ttest(barPoints(:,sortBars(x)), barPoints(:,sortBars(y)), 'tail', 'both');
        pvals(x,y)=p;
        tMatrix(x,y)=stats.tstat;
        %         tMatrix(x,y)=p;
    end
end
tMatrix(tMatrix>200)=200;
tMatrix(tMatrix<-200)=-200;

%This matrix uses 10*10 pixel cells, because some viewing software
%interpolates pixel edges
tMatrixImg=zeros(size(tMatrix,1)*10, size(tMatrix,2)*10);
for x=1:size(tMatrix,1)
    for y=1:size(tMatrix,1)
        tMatrixImg((x-1)*10+1:x*10, (y-1)*10+1:y*10)=tMatrix(x,y);
    end
end
tMatrixImg(isnan(tMatrixImg))=0;
%Add image of resulting t-statistics
%figure;
% subplot(length(numROIs),2,2*(nr));

subplot(1,2,2)
imagesc(tMatrixImg(end-59:end,end-59:end));
% imagesc(tMatrixImg);
clim = [-10 10];
ax=gca;
ax.XTickLabelRotation=90;
ax.XTick = [10:10:60];
set(ax,'xticklabel',modelNamesSort(end-5:end))
% set(ax,'xticklabel',modelNamesSort(sortBars))
ax.YTick = [10:10:60];
set(ax,'yticklabel',modelNamesSort(end-5:end))
% set(ax,'yticklabel',modelNamesSort(sortBars))
colormap(coolhotCmap([0], [128]));
colorbar;
axis image
% end
saveas(gcf, ['ModelComparisonAllROIsCombined'], 'epsc');


%% Per map/hemi/odd-even
miniMeans = barDataMeans;
tmp = miniMeans(:,:,:,:,1); tmp = tmp(:); tmp(isnan(tmp)) = 0; tmpPoints(:,1) = tmp;  
tmpMeans(1) = mean(tmp);
tmp = miniMeans(:,:,:,:,10); tmp = tmp(:); tmp(isnan(tmp)) = 0; tmpPoints(:,2) = tmp;
tmpMeans(2) = mean(tmp);
[h,p,ci,s] = ttest(tmpPoints(:,1),tmpPoints(:,2),'tail','both');
plist(1,1) = p;
plist(1,2) = s.tstat;
plist(1,3) = s.df;
% tmpMeans

for n = 1:length(whichBars)
    tmp = miniMeans(:,:,:,:,whichBars(n)); tmp = tmp(:); tmp(isnan(tmp)) = 0; tmpPoints(:,n) = tmp;  
    tmpMeans(n) = mean(tmp);
end
    tmpMeans
%%    
% Mean per participant
newMeans = miniMeans;
newMeans(isnan(newMeans)) = 0;
newMeans = squeeze(mean(newMeans,4));
newMeans = squeeze(mean(newMeans,3));
newMeans = squeeze(mean(newMeans,2));


inds = whichBars(end-1);
logmodelMean = newMeans(:,inds);
linmodelMean = newMeans(:,whichBars(end));


[h,p,ci,s] = ttest(logmodelMean,linmodelMean,'tail','both');
plist(1,1) = p;
plist(1,2) = s.tstat;
plist(1,3) = s.df;

%% Mean per hemisphere
newMeans = miniMeans;
newMeans(isnan(newMeans)) = 0;
newMeans = squeeze(mean(newMeans,4));
% newMeans = squeeze(mean(newMeans,3));
newMeans = squeeze(mean(newMeans,2));


inds = whichBars(end-1);
logmodelMean = newMeans(:,:,inds); logmodelMean = logmodelMean(:);
linmodelMean = newMeans(:,:,whichBars(end)); linmodelMean = linmodelMean(:);

[h,p,ci,s] = ttest(logmodelMean,linmodelMean,'tail','both');
plist(1,1) = p;
plist(1,2) = s.tstat;
plist(1,3) = s.df;
%%
% Per hemi/odd-even, collapsed over odd-even
miniMeans = barDataMeans;
miniMeans = squeeze(mean(barDataMeans,4)); % reduce over maps
tmp = miniMeans(:,:,:,1); tmp = tmp(:); tmp = tmp(~isnan(tmp)); tmpPoints = tmp; %lin2dOvaldurperfreeexp
tmpMeans(1) = mean(tmp);
tmp = miniMeans(:,:,:,14); tmp = tmp(:); tmp = tmp(~isnan(tmp)); tmpPoints(:,2) = tmp; %log2dOvaldurperfreeexp
tmpMeans(2) = mean(tmp);
[h,p,ci,s] = ttest(tmpPoints(:,1),tmpPoints(:,2),'tail','both');
plist(2,1) = p;
plist(2,2) = s.tstat;
plist(2,3) = s.df;

% Per odd-even, collapsed over maps and hemi's
miniMeans = barDataMeans;
miniMeans = squeeze(mean(barDataMeans,2)); % reduce over maps
miniMeans = squeeze(mean(miniMeans,2)); % reduce over hemi's
% miniMeans = squeeze(mean(barDataMeans,2)); % DO NOT reduce over odd-even
tmp = miniMeans(:,:,1); tmp = tmp(:); tmp = tmp(~isnan(tmp)); tmpPoints = tmp; %lin2dOvaldurperfreeexp
tmpMeans(1) = mean(tmp);
tmp = miniMeans(:,:,14); tmp = tmp(:); tmp = tmp(~isnan(tmp)); tmpPoints(:,2) = tmp; %log2dOvaldurperfreeexp
tmpMeans(2) = mean(tmp);
[h,p,ci,s] = ttest(tmpPoints(:,1),tmpPoints(:,2),'tail','both');
plist(3,1) = p;
plist(3,2) = s.tstat;
plist(3,3) = s.df;


% Per odd-even, collapsed over maps and hemi's
miniMeans = barDataMeans;
miniMeans = squeeze(mean(barDataMeans,2)); % reduce over maps
miniMeans = squeeze(mean(miniMeans,3)); % reduce over odd-even
% miniMeans = squeeze(mean(barDataMeans,2)); % reduce over odd-even
tmp = miniMeans(:,:,1); tmp = tmp(:); tmp = tmp(~isnan(tmp)); tmpPoints = tmp; %lin2dOvaldurperfreeexp
tmpMeans(1) = mean(tmp);
tmp = miniMeans(:,:,14); tmp = tmp(:); tmp = tmp(~isnan(tmp)); tmpPoints(:,2) = tmp; %log2dOvaldurperfreeexp
tmpMeans(2) = mean(tmp);
[h,p,ci,s] = ttest(tmpPoints(:,1),tmpPoints(:,2),'tail','both')
plist(4,1) = p;
plist(4,2) = s.tstat;
plist(4,3) = s.df;

%%
%Histograms of duration and period preferences
%subjectOrder=["dataS8", "dataS9", "dataS10", "dataS11", "dataS12", "dataS13", "dataHarv", "dataNell"];
subjectOrder={'dataAllSub'};
% mapNames={'AnteriorBelt','MiddleBelt','PosteriorBelt','Premotor'};%Revision
hemispheres={'Left', 'Right'};
modelNames=modelNamesAll;

%Get data locations
veThresh=0.1;
whichSub=1;
% figure; hold on;
for whichModel = m%1:length(modelNamesAll)
    figure; hold on;
    for whichMap=1:length(mapNames)
        for whichHemi=1:2;
            data=eval(strcat(subjectOrder{whichSub}, '.', mapNames{whichMap}, '.', hemispheres{whichHemi}, '.', modelNames{whichModel} ,'.All'));
            subplot(length(hemispheres),length(mapNames),(whichHemi-1)*length(mapNames)+whichMap);
            histogram(data.x0s(data.x0s>=0.1 & data.x0s<=1.05 & data.ves>veThresh), 0.1:0.05:2, 'FaceColor', 'blue');
            hold on;
            histogram(data.y0s(data.y0s>=0.1 & data.y0s<=1.05 & data.ves>veThresh), 0.1:0.05:2, 'FaceColor', 'red');
            axis square;
            title(strcat(hemispheres(whichHemi), ' ', mapNames(whichMap)))
            legend('duration', 'period')
            
            
        end
    end
    sgtitle(modelNamesAll(whichModel))
end

%% mean of means of duration and period preferences
%subjectOrder=["dataS8", "dataS9", "dataS10", "dataS11", "dataS12", "dataS13", "dataHarv", "dataNell"];
% subjectOrder={'dataAllSub'};
% mapNames={'AnteriorBelt','MiddleBelt','PosteriorBelt','Premotor'};%Revision
hemispheres={'Left', 'Right'};
modelNames=modelNamesAll;
m = 1
%Get data locations
veThresh=0.1;
for whichSub = 1:6
for whichModel = m%1:length(modelNamesAll)
    for whichMap=1:length(mapNames)
        for whichHemi=1:2;
            data=eval(strcat(subjectOrder{whichSub}, '.', mapNames{whichMap}, '.', hemispheres{whichHemi}, '.', modelNames{whichModel} ,'.All'));
            tmp1 = data.x0s>=0.1 & data.x0s<=1.1 & data.ves>veThresh;
            tmp2 = data.y0s>=0.1 & data.y0s<=1.1 & data.ves>veThresh;
            inds = tmp1 & tmp2;
            tmp3 = data.x0s(inds);
            tmp4 = data.y0s(inds);
            xmeans(whichSub, whichMap, whichHemi) = mean(tmp3);
            ymeans(whichSub, whichMap, whichHemi) = mean(tmp4);
        end
        end
end
end

% roimeans = mean(xmeans,3)
% roimeans = mean(roimeans,1)
% roimeans = mean(roimeans)

xmean = mean(xmeans(:))
ymean = mean(ymeans(:))
%% Duration Preferences versus Tuning Widths
subjectOrder={'dataS5','dataS2','dataS6','dataS1','dataS4','dataS3'};%, 'dataAllSub'};
% subjectOrder={'dataAllSub'};
% mapNames={'AnteriorBelt','MiddleBelt','PosteriorBelt','Premotor'};%Revision
hemispheres={'Left', 'Right'};
loadallnames;
modelNames = DurationModelNames;

%Get data locations
veThresh=0.1;
% whichSub=1;
% figure; hold on;
for whichSub = 1:6
    for whichModel = 1%1:length(modelNamesAll)
        figure; hold on;
        for whichMap=1:length(mapNames)
            for whichHemi=1:2;
                data=eval(strcat(subjectOrder{whichSub}, '.', mapNames{whichMap}, '.', hemispheres{whichHemi}, '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All'));
                subplot(length(hemispheres),length(mapNames),(whichHemi-1)*length(mapNames)+whichMap);
                dur = data.x0s(data.x0s >= 0.05 & data.x0s <= 2 & data.y0s >= 0.05 & data.y0s <= 2 & data.ves>=veThresh);
                sig = data.sigmas(data.x0s >= 0.05 & data.x0s <= 2 & data.y0s >= 0.05 & data.y0s <= 2 & data.ves>=veThresh);
                plot(dur,sig,'b.');
                hold on;
                %             plot(data.x0s(data.x0s>=0.1 & data.x0s<=2 & data.ves>veThresh),data.sigmaMinor(data.x0s>=0.1 & data.x0s<=2 & data.ves>veThresh),'.');
                %             plot(data.x0s(data.x0s>=0.1 & data.x0s<=2 & data.ves>veThresh),data.sigmaTheta(data.x0s>=0.1 & data.x0s<=2 & data.ves>veThresh),'x');
                per = data.y0s(data.x0s >= 0.05 & data.x0s <= 2 & data.y0s >= 0.05 & data.y0s <= 2 & data.ves>=veThresh);
                plot(dur,per,'r.');
                axis square;
                xlabel('duration in seconds')
                title(strcat(hemispheres(whichHemi), ' ', mapNames(whichMap)))
                legend('tuning widths', 'period')
            end
        end
        sgtitle(['Lin2dOvalGaussianDurationPeriodexpIntensityfree-',subjectOrder{whichSub}])
    end
end
%%
% subjectOrder={'dataAllSub'};
subjectOrder={'dataS5','dataS2','dataS6','dataS1','dataS4','dataS3'};
% subname = {'S5', 'S2', 'S6', 'S1', 'S5', 'S3'};;
mapNames={'AnteriorBelt','MiddleBelt','PosteriorBelt','Premotor'};%,'All'};%Revision
hemispheres={'Left', 'Right'};

%Get data locations
veThresh=0.1;
whichSub=1;
% figure; hold on;
for whichSub = 1:6
    for whichModel = 1%whichBars(end-2:end)
        figure; hold on;
        for whichMap=1:length(mapNames)
            for whichHemi=1:2;
                data=eval(strcat(subjectOrder{whichSub}, '.', mapNames{whichMap}, '.', hemispheres{whichHemi}, '.', modelNames{whichModel} ,'.All'));
                subplot(length(hemispheres),length(mapNames),(whichHemi-1)*length(mapNames)+whichMap);
                if whichModel==whichBars(end-1)
                    %                 plot([exp((data.x0s(data.x0s>0.05 & data.x0s<2 & data.ves>veThresh))-3)],[exp((data.y0s(data.x0s>0.05 & data.x0s<2 & data.ves>veThresh))-3)], '.');
                    plot([data.x0s(data.x0s>=0.1 & data.x0s<=2 & data.ves>veThresh)],[exp((data.y0s(data.x0s>=0.1 & data.x0s<=2 & data.ves>veThresh))-3)], '.');
                else
                    plot([data.x0s(data.x0s>=0.1 & data.x0s<=2 & data.ves>veThresh)],[data.y0s(data.x0s>=0.1 & data.x0s<=2 & data.ves>veThresh)], '.');
                    axis([0,1,0,1])
                end
                axis square;
                title(strcat(hemispheres(whichHemi), ' ', mapNames(whichMap)))
                xlabel('duration'); ylabel('period');
            end
        end
        sgtitle([modelNamesAll(whichModel),subjectOrder{whichSub}])
    end
end

%%
%ANOVAs
subOrder= subjectOrder;

modelNamesAll =     {'Lin2dOvalGaussianDurationPeriodexpIntensityfree',
    'LinlinearXcompressiveYNoNormOccupancyDurFreq20'      ,
    'LinlinearXlinearYNoNormOccupancyDurFreq20'           ,
    'LincompressiveXcompressiveYNoNormOccupancyDurFreq20' ,
    'LinFlatFreqOnly20'                                   ,
    'Log2dCircGaussianOccupancyPeriodlogIntensity1'       ,
    'Log2dOvalGaussianOccupancyPeriodlogIntensity1'       ,
    'Lin2dOvalGaussianOccupancyPeriodexpIntensity1'       ,
    'Lin2dOvalGaussianOccupancyPeriodexpIntensityfree'    ,
    'Log2dOvalGaussianDurationPeriodexpIntensityfree'     ,
    'Lin1dGaussianXcompressiveYDurFreq20'                 ,
    'Lin2dCircGaussianDurationPeriodexpIntensity1'     ,
    'Lin2dCircGaussianDurationPeriodexpIntensityfree',
    'Tonotopy'};

modelNamesSort;
mapNames = mapNames;
mapNames = {'HeschlsGyrus', 'AnteriorBelt', 'MiddleBelt', 'PosteriorBelt', 'Premotor'};
hemispheres={'Left', 'Right'};
x = 1;

VEs=nan([length(subjectOrder) length(mapNames) length(hemispheres)]);
Exps=VEs;
SigmaMajor=VEs;
SigmaMinor=VEs;
SigmaRatio=VEs;
SigmaTheta=VEs;
Q1D=VEs;
Q2D=VEs;
Q3D=VEs;
IQRD=VEs;
Q1P=VEs;
Q2P=VEs;
Q3P=VEs;
IQRP=VEs;
linearSlopeMajor=VEs;
vSlopeMajor=VEs;
linearSlopeMinor=VEs;
vSlopeMinor=VEs;
nVoxels=VEs;

TonotopyVEs = VEs;

% figure; hold on;
for n=1:length(subjectOrder)
    for m=1:length(mapNames)
        for whichHemi=1:length(hemispheres)
            if eval(char(strcat('isfield(', subjectOrder{n}, ', ''', mapNames{m}, ''') && isfield(', subjectOrder{n}, '.', mapNames{m}, ',''',hemispheres{whichHemi}, ''')' )))
                
                eval(char(strcat('data=', subjectOrder{n},'.', mapNames{m},'.', hemispheres{whichHemi},'.', modelNamesAll{1}  ,'.All',';')));
                
%                 veIndices=data.ves>0.1 & data.x0s>0.1 & data.x0s<1 ;
                   veIndices = data.ves > 0.1;
                if any(veIndices)
                    nVoxels(n,m,whichHemi)=sum(veIndices);
                end
                VEs(n,m,whichHemi)=mean(data.ves(veIndices));
                try
                Exps(n,m,whichHemi)=mean(data.exp(veIndices));
                catch
                    % Nothing, model without Exps
                end
                SigmaMajor(n,m,whichHemi)=mean(data.sigmas(veIndices));
                SigmaMinor(n,m,whichHemi)=mean(data.sigmaMinor(veIndices));
                SigmaRatio(n,m,whichHemi)=mean(data.sigmas(veIndices)./data.sigmaMinor(veIndices));
                SigmaTheta(n,m,whichHemi)=mean(data.sigmaTheta(veIndices));
                Q1D(n,m,whichHemi)=prctile(data.x0s(veIndices), 25);
                Q2D(n,m,whichHemi)=mean(data.x0s(veIndices));
                Q3D(n,m,whichHemi)=prctile(data.x0s(veIndices), 75);
                IQRD(n,m,whichHemi)=prctile(data.x0s(veIndices), 75)-prctile(data.x0s(veIndices), 25);
                Q1P(n,m,whichHemi)=prctile(data.y0s(veIndices), 25);
                Q2P(n,m,whichHemi)=mean(data.y0s(veIndices));
                Q3P(n,m,whichHemi)=prctile(data.y0s(veIndices), 75);
                IQRP(n,m,whichHemi)=prctile(data.y0s(veIndices), 75)-prctile(data.y0s(veIndices), 25);
                
            end
            
            % Add tonotopy VEs for comparison
            if eval(char(strcat('isfield(', subjectOrder{n}, ', ''', mapNames{m}, ''') && isfield(', subjectOrder{n}, '.', mapNames{m}, ',''',hemispheres{whichHemi}, ''')' )))
                eval(char(strcat('data=', subjectOrder{n},'.', mapNames{m},'.', hemispheres{whichHemi},'.', modelNamesAll{end}  ,'.All',';')));
                veIndices = data.ves > 0.2;
                TonotopyVEs(n,m,whichHemi)=mean(data.ves(veIndices));
            end
            
            
            
            
        end
    end
end

%Setting up ANOVA structures
hemisphereGroups=cat(3, ones(size(VEs(:,:,1))), ones(size(VEs(:,:,1)))*2);
tmp=1:length(subjectOrder);
subjectLabels=cat(3, repmat(tmp(:), [1,length(mapNames)]), repmat(tmp(:), [1,length(mapNames)]));
mapLabels=cat(3, repmat(1:length(mapNames), [length(subjectOrder),1]), repmat(1:length(mapNames), [length(subjectOrder),1]));
mapLabels=mapLabels(:);
subjectLabels=subjectLabels(:);
hemisphereLabels=hemisphereGroups(:);
subjectLabels=subjectOrder(subjectLabels);
mapLabels=mapNames(mapLabels);
hemiTmp={'L', 'R'};
% hemisphereLabels=hemiTmp(hemisphereLabels);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LME instead of ANOVA, apply to same list as all ANOVA's below

%create the data table, here I changed eccentricity from 0/1 to 1/2 here but I don't think it makes a difference
% fig2_data_lme=table(veANOVA,nominal(subjectANOVA),nominal(roiANOVA),nominal(eccANOVA+1),'VariableNames',{'ve', 'subject' ,'map', 'eccentricity'});



numSubs = repmat([1:6],[1,10]);
numMaps = repmat([1:5],[6,2]);
% fig2_data_lme=table(VEs(:),nominal(numSubs(:)),nominal(numMaps(:)), nominal(hemisphereLabels(:)), nominal(Q2D(:)),'VariableNames',{'ve', 'subject' ,'map', 'hemisphere', 'exponent'});
% fig2_data_lme=table(VEs(:),nominal(numSubs(:)), nominal(numMaps(:)), nominal(Q2D(:)),'VariableNames',{'ve', 'subject', 'map', 'exponent'});
fig2_data_lme=table(Exps(:),nominal(numSubs(:)),nominal(numMaps(:)),'VariableNames',{'exponent', 'subject' ,'map'});

%pass the table into fitlme
%random factor is specified with (1|subject), 1 denotes intercept
%there are no interaction terms in this model just main effects, estimator is restricted maximum likelihood
% fig2_anova_lme= fitlme(fig2_data_lme,'ve~map+exponent+hemisphere+(1|subject)','DummyVarCoding','effects','FitMethod','reml')
% fig2_anova_lme= fitlme(fig2_data_lme,'ve~1+map+(1|subject)','DummyVarCoding','effects','FitMethod','reml');
% fig2_anova_lme= fitlme(fig2_data_lme,'ve~map+exponent+(1|subject)','DummyVarCoding','reference','FitMethod','reml')
fig2_anova_lme= fitlme(fig2_data_lme,'exponent~map+(1|subject)','DummyVarCoding','effects','FitMethod','reml')

%pass the lme results into anova to get df corrections for reporting combined main effects
anova(fig2_anova_lme,'dfmethod','satterthwaite')

% nVox
fig2_data_lme=table(nVoxels(:),nominal(numSubs(:)),nominal(numMaps(:)),'VariableNames',{'nVoxels', 'subject' ,'map'});
fig2_anova_lme= fitlme(fig2_data_lme,'nVoxels~map+(1|subject)','DummyVarCoding','effects','FitMethod','reml')
anova(fig2_anova_lme,'dfmethod','satterthwaite')
% VE
fig2_data_lme=table(VEs(:),nominal(numSubs(:)),nominal(numMaps(:)),'VariableNames',{'ve', 'subject' ,'map'});
fig2_anova_lme= fitlme(fig2_data_lme,'ve~map+(1|subject)','DummyVarCoding','effects','FitMethod','reml')
anova(fig2_anova_lme,'dfmethod','satterthwaite')
% MeanD
fig2_data_lme=table(Q2D(:),nominal(numSubs(:)),nominal(numMaps(:)),'VariableNames',{'meanDuration', 'subject' ,'map'});
fig2_anova_lme= fitlme(fig2_data_lme,'meanDuration~map+(1|subject)','DummyVarCoding','effects','FitMethod','reml')
anova(fig2_anova_lme,'dfmethod','satterthwaite')
% IQRD
fig2_data_lme=table(IQRD(:),nominal(numSubs(:)),nominal(numMaps(:)),'VariableNames',{'IQRD', 'subject' ,'map'});
fig2_anova_lme= fitlme(fig2_data_lme,'IQRD~map+(1|subject)','DummyVarCoding','effects','FitMethod','reml')
anova(fig2_anova_lme,'dfmethod','satterthwaite')
% MeanP
fig2_data_lme=table(Q2P(:),nominal(numSubs(:)),nominal(numMaps(:)),'VariableNames',{'meanPeriod', 'subject' ,'map'});
fig2_anova_lme= fitlme(fig2_data_lme,'meanPeriod~map+(1|subject)','DummyVarCoding','effects','FitMethod','reml')
anova(fig2_anova_lme,'dfmethod','satterthwaite')
% IQRP
fig2_data_lme=table(IQRP(:),nominal(numSubs(:)),nominal(numMaps(:)),'VariableNames',{'IQRP', 'subject' ,'map'});
fig2_anova_lme= fitlme(fig2_data_lme,'IQRP~map+(1|subject)','DummyVarCoding','effects','FitMethod','reml')
anova(fig2_anova_lme,'dfmethod','satterthwaite')
% Major
fig2_data_lme=table(SigmaMajor(:),nominal(numSubs(:)),nominal(numMaps(:)),'VariableNames',{'SigmaMajor', 'subject' ,'map'});
fig2_anova_lme= fitlme(fig2_data_lme,'SigmaMajor~map+(1|subject)','DummyVarCoding','effects','FitMethod','reml')
anova(fig2_anova_lme,'dfmethod','satterthwaite')
% Minor
fig2_data_lme=table(SigmaMinor(:),nominal(numSubs(:)),nominal(numMaps(:)),'VariableNames',{'SigmaMinor', 'subject' ,'map'});
fig2_anova_lme= fitlme(fig2_data_lme,'SigmaMinor~map+(1|subject)','DummyVarCoding','effects','FitMethod','reml')
anova(fig2_anova_lme,'dfmethod','satterthwaite')
% Ratio
fig2_data_lme=table(SigmaRatio(:),nominal(numSubs(:)),nominal(numMaps(:)),'VariableNames',{'ratio', 'subject' ,'map'});
fig2_anova_lme= fitlme(fig2_data_lme,'ratio~map+(1|subject)','DummyVarCoding','effects','FitMethod','reml')
anova(fig2_anova_lme,'dfmethod','satterthwaite')
% Exp
fig2_data_lme=table(Exps(:),nominal(numSubs(:)),nominal(numMaps(:)),'VariableNames',{'exponent', 'subject' ,'map'});
fig2_anova_lme= fitlme(fig2_data_lme,'exponent~map+(1|subject)','DummyVarCoding','effects','FitMethod','reml')
anova(fig2_anova_lme,'dfmethod','satterthwaite')

% Theta
fig2_data_lme=table(SigmaTheta(:),nominal(numSubs(:)),nominal(numMaps(:)),'VariableNames',{'theta', 'subject' ,'map'});
fig2_anova_lme= fitlme(fig2_data_lme,'theta~map+(1|subject)','DummyVarCoding','effects','FitMethod','reml')
anova(fig2_anova_lme,'dfmethod','satterthwaite')
% Tonotopy VE
fig2_data_lme=table(TonotopyVEs(:),nominal(numSubs(:)),nominal(numMaps(:)),'VariableNames',{'ve', 'subject' ,'map'});
fig2_anova_lme= fitlme(fig2_data_lme,'ve~map+(1|subject)','DummyVarCoding','effects','FitMethod','reml')
anova(fig2_anova_lme,'dfmethod','satterthwaite')



%% Exponent
mn = {'HG','ATA','ATM','ATP','ATPM'};
% Stats and plot of exponents (general code for 3 factors, use when
% hemisphere difference)
% [p, tmp, statsOut] = anovan(Exps(:),{hemisphereLabels subjectLabels mapLabels}, 'varnames', {'hemisphere', 'subject', 'map'});
[p, tmp, statsOut] = anovan(Exps(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});
%Multiple comparison tests on ANOVA output

[results]=multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = flipud(CIs);
figure; plot(means, 'ok', 'MarkerFaceColor', 'k')
hold on;
% plot(means(:,1)-means(:,2), '.k')
% plot(means(:,1)+means(:,2), '.k')
for n = 1:5
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
title(['Exponent'])
axis([0.5 5.5 0.5 1])
axis square
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;
saveas(gcf, ['Exponent'], 'epsc');

%% Theta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIGNIFICANT BETWEEN MAPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[p, tmp, statsOut] = anovan(SigmaTheta(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});
%Multiple comparison tests on ANOVA output
% figure;
[results] =multcompare(statsOut, 'Dimension', [2]);

handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = (pi/2)-flipud(CIs);
figure; plot((pi/2)-means, 'ok', 'MarkerFaceColor', 'k')
hold on;
% plot(means(:,1)-means(:,2), '.k')
% plot(means(:,1)+means(:,2), '.k')
for n = 1:5
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
title(['Theta'])
axis([0.5 5.5 0.436 0.960])
axis square

% add ROI names
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;

% y ticks in degrees
yticks([0.436, 0.524, 0.611, 0.698, 0.785, 0.873, 0.960]) %25 degrees
yticklabels([25, 30, 35, 40, 45, 50, 55])

saveas(gcf, ['Theta'], 'epsc');
% axis([0 pi/2 0.5 length(subjectLabels)/2+0.5])


%% Variance Explained
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIGNIFICANT BETWEEN MAPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot of VEs
[p, tmp, statsOut] = anovan(VEs(:),{mapLabels subjectLabels }, 'varnames', {'map', 'subject'});
%Multiple comparison tests on ANOVA output
% figure; results=multcompare(statsOut, 'Dimension', [1])
[results] =multcompare(statsOut, 'Dimension', [1]);

handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = flipud(CIs);
figure; plot(means, 'ok', 'MarkerFaceColor', 'k')
hold on;
% plot(means(:,1)-means(:,2), '.k')
% plot(means(:,1)+means(:,2), '.k')
for n = 1:5
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
title(['Variance Explained'])
axis([0.5 5.5 0.1 0.35])
axis square
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;
saveas(gcf, ['Variance Explained'], 'epsc');
% axis([0 0.6 0.5 5.5]);%length(subjectLabels)/2+0.5])
% axis square


%% Variance Explained
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIGNIFICANT BETWEEN MAPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot of VEs
[p, tmp, statsOut] = anovan(TonotopyVEs(:),{mapLabels subjectLabels }, 'varnames', {'map', 'subject'});
%Multiple comparison tests on ANOVA output
[results] =multcompare(statsOut, 'Dimension', [1]);

handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = flipud(CIs);
figure; plot(means, 'ok', 'MarkerFaceColor', 'k')
hold on;
for n = 1:5
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
title(['Variance Explained'])
axis([0.5 5.5 0 1])
axis square
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;
saveas(gcf, ['Tonotopy Variance Explained'], 'epsc');
% axis([0 0.6 0.5 5.5]);%length(subjectLabels)/2+0.5])
% axis square

%% Size in nVoxels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIGNIFICANT BETWEEN MAPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% But that's nothing special, as there are empty maps
%Plot of ROI voxel count
% [p, tmp, statsOut] = anovan(nVoxels(:),{hemisphereLabels subjectLabels mapLabels}, 'varnames', {'hemisphere', 'subject', 'map'});
% %Multiple comparison tests on ANOVA output
% figure; results=multcompare(statsOut, 'Dimension', [1 3])
% axis([0 800 0.5 20.5])
% axis square

[p, tmp, statsOut] = anovan(nVoxels(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});%Multiple comparison tests on ANOVA output

[results] =multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = flipud(CIs);
figure; plot(means, 'ok', 'MarkerFaceColor', 'k')
hold on;

% plot(means(:,1)-means(:,2), '.k')
% plot(means(:,1)+means(:,2), '.k')
for n = 1:5
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
title(['number of Voxels'])
axis([0.5 5.5 0 500])
axis square
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;
saveas(gcf, ['number of Voxels'], 'epsc');
% figure; results=multcompare(statsOut, 'Dimension', [2])
% axis([-100 800 0.5 5.5]);%length(subjectLabels)/2+0.5])
% axis square
%% Sigma Majors
% [p, tmp, statsOut] = anovan(SigmaMajor(:),{hemisphereLabels subjectLabels mapLabels}, 'varnames', {'hemisphere', 'subject', 'map'});
%Stats, because there is no hemisphere effect
[p, tmp, statsOut] = anovan(SigmaMajor(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});
[results] =multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = flipud(CIs);
figure; plot(means, 'ok', 'MarkerFaceColor', 'k')
hold on;

% plot(means(:,1)-means(:,2), '.k')
% plot(means(:,1)+means(:,2), '.k')
for n = 1:5
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
title(['Sigma Majors'])
axis([0.5 5.5 -0.5 1.3])
axis square
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;
saveas(gcf, ['Sigma Majors'], 'epsc');
% %Multiple comparison tests on ANOVA output
% figure; results=multcompare(statsOut, 'Dimension', [2])
% axis([0.5 1.3 0.5 5.5]); %length(subjectLabels)/2+0.5])
% axis square

%% Sigma Minors
% [p, tmp, statsOut] = anovan(SigmaMinor(:),{hemisphereLabels subjectLabels mapLabels}, 'varnames', {'hemisphere', 'subject', 'map'});
[p, tmp, statsOut] = anovan(SigmaMinor(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});
[results] =multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = flipud(CIs);
figure; plot(means, 'ok', 'MarkerFaceColor', 'k')
hold on;

% plot(means(:,1)-means(:,2), '.k')
% plot(means(:,1)+means(:,2), '.k')
for n = 1:5
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
title(['Sigma Minors'])
axis([0.5 5.5 0 0.5])
axis square
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;
saveas(gcf, ['Sigma Minors'], 'epsc');
%Multiple comparison tests on ANOVA output
% figure; results=multcompare(statsOut, 'Dimension', [2])
% axis([0 0.5 0.5 5.5]);%length(subjectLabels)/2+0.5])
% axis square

%% Sigma ratios
% [p, tmp, statsOut] = anovan(SigmaRatio(:),{hemisphereLabels subjectLabels mapLabels}, 'varnames', {'hemisphere', 'subject', 'map'});
%Stats, because there is no hemisphere effect
[p, tmp, statsOut] = anovan(SigmaRatio(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});
[results] =multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = flipud(CIs);
figure; plot(means, 'ok', 'MarkerFaceColor', 'k')
hold on;

% plot(means(:,1)-means(:,2), '.k')
% plot(means(:,1)+means(:,2), '.k')
for n = 1:5
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
title(['Sigma Ratios'])
axis([0.5 5.5 1 7])
axis square
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;
saveas(gcf, ['Sigma Ratios'], 'epsc');
%Multiple comparison tests on ANOVA output
% figure; results=multcompare(statsOut, 'Dimension', [2])
% axis([1 10 0.5 5.5]);%length(subjectLabels)/2+0.5])
% axis square
% %Stats, because there is no hemisphere effect
% [p, tmp, statsOut] = anovan(SigmaRatio(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIGNIFICANT BETWEEN MAPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Plot of sigma thetas
% [p, tmp, statsOut] = anovan(0.5*pi-SigmaTheta(:),{hemisphereLabels subjectLabels mapLabels}, 'varnames', {'hemisphere', 'subject', 'map'});
[p, tmp, statsOut] = anovan(SigmaTheta(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});
[results] =multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = flipud(CIs);
figure; plot(means, 'ok', 'MarkerFaceColor', 'k')
hold on;

% plot(means(:,1)-means(:,2), '.k')
% plot(means(:,1)+means(:,2), '.k')
for n = 1:5
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
title(['Sigma Thetas'])
axis([0.5 5.5 1/8*pi 3/8*pi])
axis square
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;
saveas(gcf, ['Sigma Thetas'], 'epsc');
%Multiple comparison tests on ANOVA output
% figure; results=multcompare(statsOut, 'Dimension', [2])
% axis([1/8*pi 3/8*pi 0.5 5.5]); %length(subjectLabels)/2+0.5])
% axis square
% set(gca,'XTick',[pi/8 2*pi/8 3*pi/8])

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIGNIFICANT BETWEEN MAPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot of duration mean
% [p, tmp, statsOut] = anovan(Q2D(:),{hemisphereLabels subjectLabels mapLabels}, 'varnames', {'hemisphere', 'subject', 'map'});
% No hemisphere effect
[p, tmp, statsOut] = anovan(Q2D(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});
[results] =multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = flipud(CIs);
figure; plot(means, 'ok', 'MarkerFaceColor', 'k')
hold on;

% plot(means(:,1)-means(:,2), '.k')
% plot(means(:,1)+means(:,2), '.k')
for n = 1:5
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
title(['Duration means'])
axis([0.5 5.5 0 1])
axis square
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;
saveas(gcf, ['Duration Means'], 'epsc');
%Multiple comparison tests on ANOVA output
% figure; results=multcompare(statsOut, 'Dimension', [2])
% axis([0 1 0.5 5.5]);%length(subjectLabels)/2+0.5])
% axis square
%%
% Period Mean
figure;
[p, tmp, statsOut] = anovan(Q2P(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});
[results] =multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = flipud(CIs);
figure; plot(means, 'ok', 'MarkerFaceColor', 'k')
hold on;


% plot(means(:,1)-means(:,2), '.k')
% plot(means(:,1)+means(:,2), '.k')
for n = 1:5
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
title(['Period means'])
axis([0.5 5.5 0 1.3])
axis square
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;
saveas(gcf, ['Period Means'], 'epsc');
%Multiple comparison tests on ANOVA output
% figure; results=multcompare(statsOut, 'Dimension', [2])
% axis([0 1 0.5 5.5]);%length(subjectLabels)/2+0.5])
% axis square

%% Duration IQR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIGNIFICANT BETWEEN MAPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[p, tmp, statsOut] = anovan(IQRD(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});
% [p, tmp, statsOut] = anovan(IQRD(:),{hemisphereLabels subjectLabels mapLabels}, 'varnames', {'hemisphere', 'subject', 'map'});
%Multiple comparison tests on ANOVA output
[results] =multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = flipud(CIs);

figure; plot(means, 'ok', 'MarkerFaceColor', 'k')
hold on;
% plot(means(:,1)-means(:,2), '.k')
% plot(means(:,1)+means(:,2), '.k')
for n = 1:5
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
title(['Duration IQR'])
axis([0.5 5.5 0 1])
axis square
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;
saveas(gcf, ['Duration IQR'], 'epsc');
% figure; results=multcompare(statsOut, 'Dimension', [2])
% axis([0 1 0.5 5.5])
% axis square
%HEMISPHERES DIFFER!
% [p, tmp, statsOut] = anovan(IQRD(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});

%% Period IQR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIGNIFICANT BETWEEN MAPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [p, tmp, statsOut] = anovan(IQRP(:),{hemisphereLabels subjectLabels mapLabels}, 'varnames', {'hemisphere', 'subject', 'map'});
%Multiple comparison tests on ANOVA output
[p, tmp, statsOut] = anovan(IQRP(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});

[results] =multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = flipud(CIs);


figure; plot(means, 'ok', 'MarkerFaceColor', 'k')
hold on;
% plot(means(:,1)-means(:,2), '.k')
% plot(means(:,1)+means(:,2), '.k')
for n = 1:5
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
title(['Period IQR'])
axis([0.5 5.5 0 1.2])
axis square
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;
saveas(gcf, ['Period IQR'], 'epsc');
% figure; results=multcompare(statsOut, 'Dimension', [2])
% axis([0 1.2 0.5 5.5])
% axis square
% ax = gca;
% ax.XTickLabelRotation = 90;


%% Duration + Peariod means in same plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIGNIFICANT BETWEEN MAPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot of duration mean
% [p, tmp, statsOut] = anovan(Q2D(:),{hemisphereLabels subjectLabels mapLabels}, 'varnames', {'hemisphere', 'subject', 'map'});
% No hemisphere effect
[p, tmp, statsOut] = anovan(Q2D(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});
[resultsd] =multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; meansd((n+1)/2)=dataObjs(n+1).XData; CIsd(((n+1)/2),:)=dataObjs(n).XData; end
meansd = fliplr(meansd);
CIsd = flipud(CIsd);

[p, tmp, statsOut] = anovan(Q2P(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});
[resultsp] =multcompare(statsOut, 'Dimension', [2]);%, 'alpha', 0.05);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; meansp((n+1)/2)=dataObjs(n+1).XData; CIsp(((n+1)/2),:)=dataObjs(n).XData; end
meansp = fliplr(meansp);
CIsp = flipud(CIsp);

figure; plot(meansd, 'ok', 'MarkerFaceColor', 'k')
hold on;
plot(meansp, 'ok', 'MarkerFaceColor', 'k')
% plot(means(:,1)-means(:,2), '.k')
% plot(means(:,1)+means(:,2), '.k')
for n = 1:5
    h = line([n n], [CIsd(n,1) CIsd(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
    
    h = line([n n], [CIsp(n,1) CIsp(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
title(['Duration and Period means'])
axis([0.5 5.5 0 1.2])
axis square
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;
saveas(gcf, ['Duration and Period Means'], 'epsc');
%Multiple comparison tests on ANOVA output
% figure; results=multcompare(statsOut, 'Dimension', [2])
% axis([0 1 0.5 5.5]);%length(subjectLabels)/2+0.5])
% axis square

% handle = gcf;
% axObjs = handle.Children;
% dataObjs = axObjs.Children;
% for n=1:2:9; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end

%% Sigma Major and Sigma Minor in smae plot

%% Sigma Majors
% [p, tmp, statsOut] = anovan(SigmaMajor(:),{hemisphereLabels subjectLabels mapLabels}, 'varnames', {'hemisphere', 'subject', 'map'});
%Stats, because there is no hemisphere effect
[p, tmp, statsOut] = anovan(SigmaMajor(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});
[results] =multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; meansmaj((n+1)/2)=dataObjs(n+1).XData; CIsmaj(((n+1)/2),:)=dataObjs(n).XData; end
meansmaj = fliplr(meansmaj);
CIsmaj = flipud(CIsmaj);

[p, tmp, statsOut] = anovan(SigmaMinor(:),{subjectLabels mapLabels}, 'varnames', {'subject', 'map'});
[results] =multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:9; meansmin((n+1)/2)=dataObjs(n+1).XData; CIsmin(((n+1)/2),:)=dataObjs(n).XData; end
meansmin = fliplr(meansmin);
CIsmin = flipud(CIsmin);


figure; plot(meansmaj, 'ok', 'MarkerFaceColor', 'k')
hold on;
for n = 1:5
    h = line([n n], [CIsmaj(n,1) CIsmaj(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end
% title(['Sigma Majors'])
% axis([0.5 5.5 -0.5 1.3])
% axis square
% saveas(gcf, ['Sigma Majors'], 'epsc');
% %Multiple comparison tests on ANOVA output
% figure; results=multcompare(statsOut, 'Dimension', [2])
% axis([0.5 1.3 0.5 5.5]); %length(subjectLabels)/2+0.5])
% axis square

% Sigma Minors
plot(meansmin, 'ok', 'MarkerFaceColor', 'k')
% hold on;

for n = 1:5
    h = line([n n], [CIsmin(n,1) CIsmin(n,2)]);
    get(h)
    h.Color = [0 0 0];
    h.LineWidth = 2;
    set(h)
end

title(['Sigma Majors and Sigma Minors'])
axis([0.5 5.5 0 1.2])
xticks([1,2,3,4,5])
xticklabels(mn)
xticklabel_rotate;
axis square
saveas(gcf, ['Sigma Majors and Sigma Minors'], 'epsc');













%%
%Cartoon plot of exponent scaling
Yvals=0:0.01:10;
% exps=0.1:0.1:1;

exps = [0.5, 0.86, 0.8, 0.84, 0.78, 0.89];
figure; 
%hold on;
for count=1:length(exps)
    plot(Yvals, ((Yvals.^exps(count))));
    axis square
    ylim([0 10])
    
    
    
    saveas(gcf, ['exp=', num2str(exps(count))], 'epsc')
end

legend


%% Compare the odd even across conditions:
% types = {'All', 'Odd', 'Even'};
% for nt=1:length(types)
for whichModel = 1:length(modelNamesAll)
    %     eval(char(strcat('allx = dataAllSub.All.Left.',modelNamesSort(whichModel),'.All')));
    eval(char(strcat('oddxL = dataS1.All.Left.',modelNamesAll(whichModel),'.Odd')));
    eval(char(strcat('oddxR = dataS1.All.Right.',modelNamesAll(whichModel),'.Odd')));
    eval(char(strcat('evenxL = dataS1.All.Left.',modelNamesAll(whichModel),'.Even')));
    eval(char(strcat('evenxR = dataS1.All.Right.',modelNamesAll(whichModel),'.Even')));
    %     oddx = [oddxL,oddxR];
    %     evenx = [evenxL,evenxR];
    OddIndicesL=oddxL.ves>thr & oddxL.x0s>=0.05 & oddxL.x0s<=1;
    OddIndicesR = oddxR.ves>thr & oddxR.x0s>=0.05 & oddxR.x0s<=1 ;
    EvenIndicesL=evenxL.ves>thr & evenxL.x0s>=0.05 & evenxL.x0s<=1 ;
    EvenIndicesR=evenxR.ves>thr & evenxR.x0s>=0.05 & evenxR.x0s<=1 ;
    LeftIndices = logical(OddIndicesL.*EvenIndicesL);
    RightIndices = logical(OddIndicesR.*EvenIndicesR);
    %     veIndices=logical(OddIndices.*EvenIndices);
    
    eval(char(strcat('OddL = dataS1.All.Left.',modelNamesAll(whichModel),'.Odd.x0s(LeftIndices)')));
    eval(char(strcat('OddR = dataS1.All.Right.',modelNamesAll(whichModel),'.Odd.x0s(RightIndices)')));
    eval(char(strcat('EvenL = dataS1.All.Left.',modelNamesAll(whichModel),'.Even.x0s(LeftIndices)')));
    eval(char(strcat('EvenR = dataS1.All.Right.',modelNamesAll(whichModel),'.Even.x0s(RightIndices)')));
    Odd=[OddL,OddR];
    Even = [EvenL, EvenR];
    
    if (whichModel == 1) || (whichModel == 2) || (whichModel == 3)
        r(1,2) = 1
    else
        [r,p] = corrcoef(Even,Odd);
    end
    
    coefficients(whichModel) = r(1,2);
    n = (sum(LeftIndices)+sum(RightIndices))./(1.6.^2); %1.6 mm is scanning resolution
    p=r2p(r(1,2),n);
    %     pvalues(whichModel) = r2p(r,n);
    pvalues(whichModel)= p;
end
% end

figure; plot(coefficients, 'o'); hold on;
plot(pvalues,'*')
plot(ones(length(modelNamesAll)+1,1)*0.05, 'r');
plot(zeros(length(modelNamesAll)+1,1), 'k');
legend('Correlation Odd-Even (r)','p-value (corrected)')
axis square
ax = gca;
ax.XLim = [0,length(modelNamesAll)+1];
ax.YLim = [-0.1,1];
ax.XTick = [1:length(modelNamesAll)];
ax.XTickLabel = modelNamesAll;
ax.XTickLabelRotation = 90;
ax.YLabel.String = 'R and associated P-values of Odd/Even pairs';
% ax.Legend = {'Correlation (r)','p-values'};
% r2p(r,n)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Scatter plot stuff

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% figure;
% ppnames = {'dataAllSub'};
ppnames={'dataS5','dataS2','dataS6','dataS1','dataS4','dataS3', 'dataAllSub'};
mapNames = {'HeschlsGyrus', 'AnteriorBelt', 'MiddleBelt', 'PosteriorBelt', 'Premotor'};
titleNames = {'ATA', 'ATP', 'ATM', 'ATPM', 'HG'};
hemi = {'Left','Right'};
% nh = 2;
% pp = 1;

x = 0.05;
upsample = (3.^2); %1.6 mm is scanning resolution
% upsample = (1.6.^2); %1.6 mm is scanning resolution
allDurCors = [];
allDurPs = [];
allPerCors = [];
allPerPs = [];
            
for pp = 1:length(ppnames)
    for nh = 1:2
        cnt = 1;
        figure;
        for nm = 1:length(mapNames) %[5, 1,3,2,4]%
            
            vesIndicesO = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Odd.vesXval']) > 0.1 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Odd.x0s']) >= 0.05 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Odd.x0s']) <= 1 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Odd.y0s']) >= 0.05 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Odd.y0s']) <= 1.1;
            vesIndicesE = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Even.vesXval']) > 0.1 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Even.x0s']) >= 0.05 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Even.x0s']) <= 1 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Even.y0s']) >= 0.05 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Even.y0s']) <= 1.1;
            
            vesIndices = vesIndicesO & vesIndicesE;
            subplot(2,length(mapNames),cnt)
            OdddataD = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Odd.x0s(vesIndices)']);
            EvendataD = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Even.x0s(vesIndices)']);
            AlldataD = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(vesIndices)']);
            AlldataP = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s(vesIndices)']);
            tmp=corrcoef(OdddataD, EvendataD);
            try
                corD(cnt)=tmp(1,2);
            catch
                corD(cnt) = 1;
            end
            pD(cnt)=r2p(corD(cnt), length(OdddataD)./upsample);
            scatter(OdddataD,EvendataD, 20, [0 0 1], '.');% h = lsline;
            
            allDurCors = [allDurCors, corD(cnt)];
            allDurPs = [allDurPs, pD(cnt)];
            %     plot(OdddataD,EvendataD, 'b.')
            hold on;
            %     h.Color = [0 0 0];
            %     h.LineWidth = 2;
            xlim([0 1.0])
            ylim([0 1.0])
            xticks([0 0.2 0.4 0.6 0.8 1.0])
            xlabel(['Preferred duration odd runs (s)'])
            yticks([0 0.2 0.4 0.6 0.8 1.0])
            if cnt == 1
                ylabel(['Preferred duration even runs (s)'])
            end
            axis square
            %     AX1 = gca;
            %     xpoints1 = 0+10/length(EvendataD):10/length(EvendataD):10;
            %     bigx1 = [ones(size(EvendataD)); EvendataD]';
            %     bigy1 = OdddataD';
            %     b1 = bigx\bigy;
            %     hlslines = refline(AX1,b1);
            x1 = OdddataD;
            y1 = EvendataD;
            P1 = polyfit(x1,y1,1);
            yfit1 = polyval(P1, [min(x1) max(x1)]); %P(2).*[min(x1) max(x1)]+P(1);
            plot([min(x1) max(x1)],yfit1,'k-.');
            title({['r = ', num2str(corD(cnt), '%1.3f')], ['p = ', num2str(pD(cnt), '%1.3f')]});
            
            
            subplot(2,length(mapNames),cnt+length(mapNames))
            OdddataP = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Odd.y0s(vesIndices)']);
            EvendataP = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Even.y0s(vesIndices)']);
            scatter(OdddataP, EvendataP, 20, [1 0 0], '.');% h=lsline;
            
            
            tmp=corrcoef(OdddataP, EvendataP);
            try
                corP(cnt)=tmp(1,2);
            catch
                corP(cnt) = 1;
            end
            pP(cnt)=r2p(corP(cnt), length(OdddataP)./upsample);
            
            allPerCors = [allPerCors, corP(cnt)];
            allPerPs = [allPerPs, pP(cnt)];
            
            xlim([0 1.0])
            ylim([0 1.0])
            xticks([0 0.2 0.4 0.6 0.8 1.0])
            yticks([0 0.2 0.4 0.6 0.8 1.0])
            axis square
            %     h.Color = [0 0 0];
            %     h.LineWidth = 2;
            %     xlim([0 1.2])
            %     ylim([0 1.2])
            %     axis square
            %     title([titleNames{nm}, ' period Odd/Even'])
            AX2 = gca;
            hold on
            
            xlabel(['Preferred period odd runs (s)'])
            %     yticks([0 0.2 0.4 0.6 0.8 1.0 1.2])
            if cnt == 1
                ylabel(['Preferred period even runs (s)'])
            end
            %     x = 1.2/length(OdddataP):1.2/length(OdddataP):1.2;
            x2 = OdddataP;
            y2 = EvendataP;
            %     scatter(x,y1,25,'b','*')
            P2 = polyfit(x2,y2,1);
            %     yfit2 = P2(2)*x2+P2(1);
            yfit2 = polyval(P2, [min(x2) max(x2)]);
                hold on;
            plot([min(x2) max(x2)],yfit2,'k-.');
            %     beta = polyfit(xdat(ok,:),ydat(ok,:),1);
            %     hlslines = refline(AX,b);
            %     hlslines = refline(AX, P2);
            title({['r = ', num2str(corP(cnt), '%1.3f')], ['p = ', num2str(pP(cnt), '%1.3f')]});
            
            cnt = cnt+1;
%             saveas(gcf, ['correlation-odd-even-', hemi{nh}, '-', ppnames{pp}], 'epsc');
        end
    end
end
% [h, p, ci, stats] = ttest(allDurCors,0, 'tail', 'both');
% [h, p, ci, stats] = ttest(allPerCors,0, 'tail', 'both');
% 
% suptitle(['Odd/Even'])
% Can also do the same through polyfit
%     [d,f] = polyfit(EvendataD, OdddataD, 1);
%     xl = 0.1:0.01:1.3;
%     r = d(1) .* OdddataD + d(2);
%     r = polyval(d, xl);
%     plot(r);
%     plot(r, EvendataD, '.');  hold on;

% figure;
%  scatter(EvendataP, OdddataP, 20, [1 0 0], '.'); h=lsline;
%     xlim([0 1.2])
%     ylim([0 1.2])
%     xticks([0 0.2 0.4 0.6 0.8 1.0 1.2])
%     yticks([0 0.2 0.4 0.6 0.8 1.0 1.2])
%     axis square
%     hold on;
%     plot(x, yfit, 'r-.')

%% Correlate duration with Period


% figure;
ppnames = {'dataAllSub'};
mapNames = {'HeschlsGyrus', 'AnteriorBelt', 'MiddleBelt', 'PosteriorBelt', 'Premotor'};
titleNames = {'ATA', 'ATP', 'ATM', 'ATPM', 'HG'};
hemi = {'Left','Right'};
% nh = 2;
% pp = 1;

x = 0.05;
upsample = (6.^2); %1.6 mm is scanning resolution
% upsample = (1.6.^2); %1.6 mm is scanning resolution
allACors = [];
allAPs = [];

for pp = 1
    for nh = 1:2
        cnt = 1;
        figure;
        for nm = 1:length(mapNames)
            
            % Select only relevant indices
            vesIndicesA = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Odd.vesXval']) >= 0.1 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.Even.vesXval']) >= 0.1 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s']) >= 0.05 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s']) <= 1 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s']) >= 0.05 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s']) <= 1.1;
            
            vesIndices = vesIndicesA;
            subplot(1,5,cnt)
            AlldataD = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(vesIndices)']);
            AlldataP = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s(vesIndices)']);
            tmp=corrcoef(AlldataD, AlldataP);
            try
                corA(cnt)=tmp(1,2);
            catch
                corA(cnt) = 1;
            end
            % Make and stats for correlation plot
            pD(cnt)=r2p(corA(cnt), length(AlldataD)./upsample);
            allACors = [allACors, corA(cnt)];
            allAPs = [allAPs, pD(cnt)];
            scatter(AlldataD,AlldataP, 20, [0 0 1], '.');% h = lsline;
            hold on;
            xlim([0 1.0])
            ylim([0 1.0])
            xticks([0 0.2 0.4 0.6 0.8 1.0])
            xlabel(['Preferred duration (s)'])
            yticks([0 0.2 0.4 0.6 0.8 1.0])
            if cnt == 1
                ylabel(['Preferred Period (s)'])
            end
            axis square
            
            % Fit line
            x1 = AlldataD;
            y1 = AlldataP;
            P1 = polyfit(x1,y1,1);
            yfit1 = polyval(P1, [min(x1) max(x1)]); %P(2).*[min(x1) max(x1)]+P(1);
            plot([min(x1) max(x1)],yfit1,'k-.');
            title({['r = ', num2str(corA(cnt), '%1.3f')], ['p = ', num2str(pD(cnt), '%1.3f')]});
            cnt = cnt+1;
            
%             saveas(gcf, ['correlation-dur-per', hemi{nh}, '-', ppnames{pp}], 'epsc');
        end
    end
end

% [h, p, ci, stats] = ttest(allACors,0, 'tail', 'both');
%% Correlate duration with Tonotopy


% figure;
ppnames = {'dataAllSub'};
mapNames = {'HeschlsGyrus', 'AnteriorBelt', 'MiddleBelt', 'PosteriorBelt', 'Premotor'};
titleNames = {'ATA', 'ATP', 'ATM', 'ATPM', 'HG'};
hemi = {'Left','Right'};
% nh = 2;
% pp = 1;

x = 0.05;
upsample = (1.6.^2); %1.6 mm is scanning resolution
allACors = [];
allAPs = [];

for pp = 1
    for nh = 1:2
        cnt = 1;
        figure;
        for nm = 1:length(mapNames)
            % Select only relevant indices
            vesIndicesA = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves']) > 0.05 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s']) >= 0.05 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s']) <= 1 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s']) >= 0.05 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s']) <= 1.1;
            vesIndicesB = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Tonotopy.All.x0s']) <= 9 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Tonotopy.All.ves']) >= 0.05 &...
                eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Tonotopy.All.x0s']) >= 4.4;
            
            vesIndices = vesIndicesA & vesIndicesB;
            subplot(1,5, cnt)
            AlldataD = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s(vesIndices)']);
            AlldataP = eval([(ppnames{pp}), '.', (mapNames{nm}), '.', (hemi{nh}), '.Tonotopy.All.x0s(vesIndices)']);
            tmp=corrcoef(AlldataD, AlldataP);
            try
                corA(cnt)=tmp(1,2);
            catch
                corA(cnt) = 1;
            end
            % Make and stats for correlation plot
            pD(cnt)=r2p(corA(cnt), length(AlldataD)./25);
            allACors = [allACors, corA(cnt)];
            allAPs = [allAPs, pD(cnt)];
            scatter(AlldataD,AlldataP, 20, [0 0 1], '.');% h = lsline;
            hold on;
            xlim([0 1.0])
            ylim([4 9])
            xticks([0 0.2 0.4 0.6 0.8 1.0])
            xlabel(['Preferred duration (s)'])
%             yticks([0 0.2 0.4 0.6 0.8 1.0])
            yticks([4 5 6 7 8 9])
%             yticklabels([exp(4) exp(5) exp(6) exp(7) exp(8) exp(9)])
            yticklabels([55 150 400 1000 3000 8100])
%             ytickangle([90])
            if cnt == 1
                ylabel(['Preferred Auditory Frequency (Hz)'])
            end

            axis square
            
            % Fit line
            x1 = AlldataD;
            y1 = AlldataP;
            P1 = polyfit(x1,y1,1);
            yfit1 = polyval(P1, [min(x1) max(x1)]); %P(2).*[min(x1) max(x1)]+P(1);
            plot([min(x1) max(x1)],yfit1,'k-.');
            title({['r = ', num2str(corA(cnt), '%1.3f')], ['p = ', num2str(pD(cnt), '%1.3f')]});
            cnt = cnt+1;
            
            saveas(gcf, ['correlation-dur-tono', hemi{nh}, '-', ppnames{pp}], 'epsc');
        end
    end
end

[h, p, ci, stats] = ttest(allACors,0, 'tail', 'both');

%% Correlate Duration with Tuning Widths

%%
% Look at the distribution of Exponents within single maps (overlay the
% individual pp's on top of the whole?
nbins = 100;
thr = 0.2;
figure;
LeftIndicesAnt  = dataAllSub.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves > thr & dataAllSub.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05 & dataAllSub.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 1;
LeftIndicesMid  = dataAllSub.MiddleBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves > thr & dataAllSub.MiddleBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05 & dataAllSub.MiddleBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 1 ;
LeftIndicesPos  = dataAllSub.PosteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves > thr & dataAllSub.PosteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05 & dataAllSub.PosteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 1 ;
LeftIndicesPre  = dataAllSub.Premotor.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves > thr & dataAllSub.Premotor.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05 & dataAllSub.Premotor.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 1 ;
RightIndicesAnt  = dataAllSub.AnteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves > thr & dataAllSub.AnteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05 & dataAllSub.AnteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 1;
RightIndicesMid  = dataAllSub.MiddleBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves > thr & dataAllSub.MiddleBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05 & dataAllSub.MiddleBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 1 ;
RightIndicesPos  = dataAllSub.PosteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves > thr & dataAllSub.PosteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05 & dataAllSub.PosteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 1 ;
RightIndicesPre  = dataAllSub.Premotor.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves > thr & dataAllSub.Premotor.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05 & dataAllSub.Premotor.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 1 ;


subplot(2,3,1); histogram(dataAllSub.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(LeftIndicesAnt),nbins); title('AnteriorBelt'); axis square
hold on;
histogram(dataAllSub.AnteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(RightIndicesAnt),nbins);

subplot(2,3,2); histogram(dataAllSub.MiddleBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(LeftIndicesMid),nbins);title('MiddleBelt'); axis square
hold on;
histogram(dataAllSub.MiddleBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(RightIndicesMid),nbins);

subplot(2,3,3); histogram(dataAllSub.PosteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(LeftIndicesPos),nbins);title('PosteriorBelt'); axis square
hold on;
histogram(dataAllSub.PosteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(RightIndicesPos),nbins);

subplot(2,3,5); histogram(dataAllSub.Premotor.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(LeftIndicesPre),nbins);title('Premotor'); axis square
hold on;
histogram(dataAllSub.Premotor.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(RightIndicesPre),nbins);

% Set up for ttest,
al = dataAllSub.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(LeftIndicesAnt);
ar = dataAllSub.AnteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(RightIndicesAnt);
ml = dataAllSub.MiddleBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(LeftIndicesMid);
mr = dataAllSub.MiddleBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(RightIndicesMid);
pl = dataAllSub.PosteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(LeftIndicesPos);
pr = dataAllSub.PosteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(RightIndicesPos);
pml = dataAllSub.Premotor.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(LeftIndicesPre);
pmr = dataAllSub.Premotor.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.exp(RightIndicesPre);

[h,p,ci,s]=ttest(al,1, 'tail', 'left'); s
[h,p,ci,s]=ttest(ar,1, 'tail', 'left'); s
[h,p,ci,s]=ttest(ml,1, 'tail', 'left'); s
[h,p,ci,s]=ttest(mr,1, 'tail', 'left'); s
[h,p,ci,s]=ttest(pl,1, 'tail', 'left'); s
[h,p,ci,s]=ttest(pr,1, 'tail', 'left'); s
[h,p,ci,s]=ttest(pml,1, 'tail', 'left'); s
[h,p,ci,s]=ttest(pmr,1, 'tail', 'left'); s

for n = 1:8
    h,p,ci,s = ttest(vs(:,n),1)
end

%%
modelNamesSort = DurationModelNames;
%Timing vs distance etc (Revision process)
% Also for Duration vs TuningWidth
subjectOrder={'dataS5','dataS2','dataS6','dataS1','dataS4','dataS3', 'dataAllSub'};
subData = subjectOrder;
mapNames={'HeschlsGyrus','AnteriorBelt','MiddleBelt','PosteriorBelt','Premotor'};%Revision
hemispheres={'Left', 'Right'};
DTnames={'All', 'Odd', 'Even'};
whichSubs=[7];
mapList = [1:5];


for thisSub=whichSubs
    clear plotdata;
    eval(['data=', char(subData(thisSub)), ';'])
    for whichMap=mapList%1:length(mapNames)
        for whichHemi=1:2
            count=(whichMap-1)*2+whichHemi;
            dataPresent=char(strcat('isfield(data, ''', mapNames{whichMap}, ''') && isfield(data.',  mapNames{whichMap}, ',''',hemispheres{whichHemi}, ''')' ));
            if eval(dataPresent)
                for whichDT=1:3;
                    dataName=char(strcat('data.', mapNames{whichMap}, '.', hemispheres{whichHemi},'.Lin2dOvalGaussianDurationPeriodexpIntensityfree.', DTnames(whichDT)));
                    eval(['dataTmp=', dataName, ';'])
                    dataTmp.ROItitle=char(strcat(subData{thisSub},mapNames{whichMap}, hemispheres{whichHemi}));
                    plotdata{count, whichDT}=dataTmp(1,1);
                end
            end
        end
    end
    %try
    %     figure; hold on;
    try
        stats{thisSub}=PlotRoiDistanceAllRoiDurationPeriod(plotdata, 0.1, [0 0 0 0 0 1 1],11); %Plot everything [1 1 1 0 1 1 1] %Tuning widths only [0 0 0 0 0 1 1]
%         saveas(gcf, ['DurationPeriod-vs-Distance-', subjectOrder{thisSub}], 'epsc');
        %end
    catch
        fprintf('\nnope\n')
        continue
    end
end

%newvec(50) = 5;, this is the frequency term in the sine equation
% compline = 0.005 : 0.995 in steps of small
%(sin(2*pi*f*t+phi) = sine(2*pi*newvec(nf).*compline+phi)
%newfitd(50) = ~1.2 cycle for 50.2 mm of cortical surface

% Make sine more densely sampled

%%
plotnames = {};
for npn = 1:4
    plotnames = [plotnames, strcat('Left', mapNames{npn})];
    plotnames = [plotnames, strcat('Right', mapNames{npn})];
end

for nn = 1:length(stats{1}.DurationSpectrum)
    for nm = 1:length(stats)
        tmpld(nn,nm) = length(stats{nm}.DurationSpectrum{nn});
        tmplp(nn,nm) = length(stats{nm}.PeriodSpectrum{nn});
    end
end
ld = max(tmpld(:));
lp = max(tmplp(:));
fullvecd = nan(6,8,ld);
fullvecp = nan(6,8,lp);

figure;
% Loop over pp and areas for duration
for nn=1:length(stats{1}.DurationSpectrum)
    subplot(2,4,nn)
    hold on;
    %     fullvec = nan(length(stats{1}.DurationSpectrum{nn}),6);
    for nm = 1:length(stats)
        tmp = 1./(stats{nm}.fft{nn}.fpoints./max(stats{nm}.fft{nn}.fpoints));
        %         plot(stats{nm}.fft{nn}.fpoints,stats{nm}.DurationSpectrum{nn})
        plot(stats{nm}.DurationSpectrum{nn})
        fullvecd(nm,nn,1:length(stats{nm}.DurationSpectrum{nn})) = stats{nm}.DurationSpectrum{nn};
        legend('1','2','3','4','5','6')
    end
    axis square
    ax = gca;
    %     L = length(stats{nm}.fft{nn}.fpoints);
    %     ax.XTick = [2:20:L];
    %     ax.XTickLabelRotation = 90;
    %     tmp2 = tmp(ax.XTick);
    %     for nt = 1:length(ax.XTick)
    %         ax.XTickLabel{nt} = num2str(tmp2(nt));
    %     end
    title(['Duration:',plotnames{nn}])
end

figure;
% Loop over pp and areas for period
for nn=1:length(stats{1}.PeriodSpectrum)
    subplot(2,4,nn)
    hold on;
    for nm = 1:length(stats)
        plot(stats{nm}.PeriodSpectrum{nn}, '.b')
        plot(stats{nm}.DurationSpectrum{nn}, '.r')
        fullvecp(nm,nn,1:length(stats{nm}.PeriodSpectrum{nn})) = stats{nm}.PeriodSpectrum{nn};
    end
    meancycle{nn,nm,1} = mean(squeeze(fullvecd(:,nn,:)),1);
    meancycle{nn,nm,2} = mean(squeeze(fullvecp(:,nn,:)),1);
    plot(meancycle{nn,nm,2}, 'b', 'LineWidth', 2);
    plot(meancycle{nn,nm,1}, 'r', 'LineWidth', 2);
    
    axis square
    ax = gca;
    title(['Period:',plotnames{nn}])
    
    
end
% figure;
% subplot(1,2,1);
% plot(stats{m}.fft{n}.fpoints,stats{m}.fft{n}.fftd);
% subplot(1,2,2);
% plot(stats{m}.fft{n}.fpoints,stats{m}.fft{n}.fftp);

%%
% is freq3 different from freq2 or freq4 etc.
%select roi
testl = min(tmpld(1,:));
testone = squeeze(fullvecd(:,1,1:testl));
[h,p,ci,smat]=ttest(testone,testone(:,end:-1:1), 'dim', 1)

% Make ANOVA struct
VEs=nan([length(subjectOrder) length(mapNames) length(hemispheres)]);


%%
ss = [1 2 3 4 5 6];
clear DurPower
clear PerPower
% sub, map, freq, hemi
n = 1;
for subs = ss%1:4%length(subjectOrder)
    for maps = 1:4%length(mapNames)
        for freq = 1:8%size(fullvecd,3)
            
            DurPower(n,maps,freq,1)= fullvecd(subs,(maps*2)-1,freq);
            DurPower(n,maps,freq,2) = fullvecd(subs,maps*2,freq);
            PerPower(n,maps,freq,1)= fullvecp(subs,(maps*2)-1,freq);
            PerPower(n,maps,freq,2) = fullvecp(subs,maps*2,freq);
            
        end
    end
    n = n+1;
end

% for hemi = 1:2

clear tvalsf pvalsf
for maps = 1:4
    for freq1 = 1:8
        for freq2 = 1:8
            %             tmp1 = squeeze(DurPower(:,maps,freq1+1,hemi));
            tmp1 = [squeeze(DurPower(:,maps,freq1,1)); squeeze(DurPower(:,maps,freq1,2))];
            tmp2 = [squeeze(DurPower(:,maps,freq2,1)); squeeze(DurPower(:,maps,freq2,2))];
            %             tmp2 = squeeze(DurPower(:,maps,freq2+1,hemi));
            [h,p,ci,smat] = ttest(tmp1,tmp2);
            tvalsf(maps,freq1,freq2) = smat.tstat;
            if ~isnan(p)
                %             if p < 0.05
                pvalsf(maps,freq1,freq2) = p;
                %             else
            else
                pvalsf(maps,freq1,freq2) = 1;
            end
        end
    end
end
% end

figure;
subplot(2,2,1);
imagesc(squeeze(abs(tvalsf(1,:,:)))); colorbar;
axis square
caxis([0 5]);
subplot(2,2,2);
imagesc(squeeze(abs(tvalsf(2,:,:)))); colorbar;
axis square
caxis([0 5]);
subplot(2,2,3);
imagesc(squeeze(abs(tvalsf(3,:,:)))); colorbar;
axis square
caxis([0 5]);
subplot(2,2,4);
imagesc(squeeze(abs(tvalsf(4,:,:)))); colorbar;
axis square
caxis([0 5]);



figure; subplot(2,2,1);
imagesc(squeeze(abs(pvalsf(1,:,:)))); colorbar;
axis square
caxis([0 1]);
subplot(2,2,2);
imagesc(squeeze(abs(pvalsf(2,:,:)))); colorbar;
axis square
caxis([0 1]);
subplot(2,2,3);
imagesc(squeeze(abs(pvalsf(3,:,:)))); colorbar;
axis square
caxis([0 1]);
subplot(2,2,4);
imagesc(squeeze(abs(pvalsf(4,:,:)))); colorbar;
axis square
caxis([0 1]);




mapNames = mapNames;
subjectOrder = subjectOrder(1:6);
hemiTmp={'L', 'R'};
for n = 1:8
    freqs{n} = num2str(n);
end

hemisphereGroups=cat(4, ones(size(DurPower(:,:,:,1))), ones(size(DurPower(:,:,:,1)))*2);
tmp=1:length(subjectOrder);
tmp2 = 1:length(freqs);
% subjectLabels=cat(4, repmat(tmp(:), [1,length(mapNames)]), repmat(tmp(:), [1,length(mapNames)]), repmat(tmp(:), [1,length(mapNames)]));
subjectLabels=repmat(tmp(:), [1,length(mapNames),length(freqs),length(hemiTmp)]);
% mapLabels=cat(4, repmat(1:length(mapNames), [length(subjectOrder),1]), repmat(1:length(mapNames), [length(subjectOrder),1]));
mapLabels=repmat(tmp(1:4), [length(subjectOrder),1,length(freqs),length(hemiTmp)]);
% WHY is this necessary? Why does repmat behave so strange here?
freqLabels=permute(repmat(tmp2(:),[1,6,4,2]),[2,3,1,4]);
mapLabels=mapLabels(:);
subjectLabels=subjectLabels(:);
hemisphereLabels=hemisphereGroups(:);
freqLabels = freqLabels(:);

subjectLabels=subjectOrder(subjectLabels);
mapLabels=mapNames(mapLabels);
hemisphereLabels=hemiTmp(hemisphereLabels);
freqLabels = freqs(freqLabels);

[subjectLabels; mapLabels; freqLabels; hemisphereLabels]

%%
% Test on freqs
% [p, tmp, statsOut] = anovan(DurPower(:),{subjectLabels  mapLabels freqLabels hemisphereLabels}, 'varnames', {'subject', 'map', 'freq', 'hemi'});%Multiple comparison tests on ANOVA output

% without hemi (NOT SIGNIFICANT)
[p, tmp, statsOut] = anovan(DurPower(:),{subjectLabels  mapLabels freqLabels}, 'varnames', {'subject', 'map', 'freq'});%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [3])

%% % And period dim, hemi not significant
[p, tmp, statsOut] = anovan(PerPower(:),{subjectLabels  mapLabels freqLabels hemisphereLabels}, 'varnames', {'subject', 'map', 'freq', 'hemi'});%Multiple comparison tests on ANOVA output

[p, tmp, statsOut] = anovan(PerPower(:),{subjectLabels  mapLabels freqLabels}, 'varnames', {'subject', 'map', 'freq'});
%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [1 2 3])

%% plot for a random subject

stats = statsAt17mm;
% stats = statsAt2mm;

figure;
ss = 4;
clear pdvals ppvals
names = {'AnteriorBeltLeft', 'AnteriorBeltRight', 'MiddleBeltLeft', 'MiddleBeltRight','PosteriorBeltLeft', 'PosteriorBeltRight','PremotorLeft', 'PremotorRight'};
for m = 1:length(stats) % Subjects
    for n = 1:length(stats{1}.coherenceAxis) % Maps
        %         tmpco = max(stats{m}.coherencesDur{n},[],2);
        tmpco = stats{m}.coherencesDur{n};
        tmpper = stats{m}.coherencesPer{n};
        %         tmpco = tmpper;
        
        
        numOfMeasures = stats{m}.bins{n};
        
        
        pdvals(m,n) = r2p(max(tmpco),numOfMeasures);
        linpdvals(m,n) = r2p(stats{m}.rLinFitX{n}, numOfMeasures);
        pdnum(m,n) = find(tmpco==max(tmpco))/4+0.5; % Expressed in cycles
        ppvals(m,n) = r2p(max(tmpper),numOfMeasures);
        linppvals(m,n) = r2p(stats{m}.rLinFitY{n}, numOfMeasures);
        ppnum(m,n) = find(tmpper==max(tmpper))/4+0.5; % Expressed in cycles
        %         tmp = fdr_bh(pdvals(m,n));
        
        
        subplot(length(stats),length(stats{1}.coherenceAxis),(m-1)*length(stats{1}.coherenceAxis)+n)
        plot(stats{m}.coherenceAxis{n},tmpco);
        hold on;
        %         plot(stats{m}.coherenceAxis{n},tmpper);
        
        tmp = zeros(1,length(tmpco))*nan;
        ind = find(tmpco==max(tmpco));
        tmp(ind) = tmpco(ind);
        
        tmpp= zeros(1,length(tmpper))*nan;
        ind2 = find(tmpper==max(tmpper));
        tmpp(ind2) = tmpper(ind2);
        
        
        plot(stats{m}.coherenceAxis{n},tmp, 'or', 'MarkerSize', 12)
        %         plot(stats{m}.coherenceAxis{n},tmpp, 'ob', 'MarkerSize', 12)
        %     figure; plot(stats{1}.meanDist{n}./stats{1}.coherenceAxis{n},stats{1}.coherencesd{n});
        xlabel('Frequency (in cycles per ROI)')
        
        ax = gca;
        %         ax.XTick = [3:ss:(stats{m}.coherenceAxis{n}(end)/(2*pi)-1)*ss+3];
        ax.XTick = [2*pi:2*pi:stats{m}.coherenceAxis{n}(end)];
        %         ax.XTickLabel = [stats{m}.coherenceAxis{n}(ax.XTick)/(2*pi)];
        ax.XTickLabel = [ax.XTick./(2*pi)];
        
        %         xlim([0,stats{m}.coherenceAxis{n}(end)])
        xlim([0,7.5*2*pi])
        ylim([-1 1]);
        if m == 1
            title(names{n})
        end
    end
end


%## Couple to nyquist max
%## Below half is useless, half to one is difficult to interpret
%## Less dense sampling, try 1/4 cycles (maybe even 1/2)
%## Fix x-axis to represent correct values (careful of the 2*pi factors).

% Show -log10(r2p(answer)); Careful, anwer is still abs here.
% fdr_bh correction.

% side by side of the linear and cyclical r+p vals per map
% include correlation for linFitDur with actual data

[a,b,c,d]=fdr_bh(pdvals,0.05,'dep');
[a2,b2,c2,d2]=fdr_bh(ppvals,0.05,'dep');
[a3,b3,c3,d3]=fdr_bh(linpdvals,0.05,'dep');
[a4,b4,c4,d4]=fdr_bh(linppvals,0.05,'dep');


figure;
subplot(2,4,1);imagesc(-log10(pdvals)); title('-log_1_0 p Duration Cycle'); colorbar; axis square;
subplot(2,4,5);imagesc(-log10(ppvals)); title('-log_1_0 p Period Cycle'); colorbar; axis square;
subplot(2,4,2);imagesc(d); caxis([0,0.05]); colorbar; axis square; title('FDR-adjusted p-val')
subplot(2,4,6);imagesc(d2); caxis([0,0.05]); colorbar; axis square; title('FDR-adjusted p-val')
subplot(2,4,3); imagesc(-log10(linpdvals)); title('-log_1_0 p Duration Linear'); colorbar; axis square;
subplot(2,4,7); imagesc(-log10(linppvals)); title('-log_1_0 p Period Linear'); colorbar; axis square;
subplot(2,4,4);imagesc(d3); caxis([0,0.05]); colorbar; axis square; title('FDR-adjusted p-val')
subplot(2,4,8);imagesc(d4); caxis([0,0.05]); colorbar; axis square; title('FDR-adjusted p-val')

%%
for m = 1:6
    for n = 1:8
        spv(m,n) = stats{m}.sin_pvals{n};
        lpv(m,n) = stats{m}.lin_pvals{n};
        spvp(m,n) = stats{m}.sinp_pvals{n};
        lpvp(m,n) = stats{m}.linp_pvals{n};
        %         stats{m}.
    end
end



[a,b,c,d]=fdr_bh(spv,0.05,'dep');
[a2,b2,c2,d2]=fdr_bh(spvp,0.05,'dep');
[a3,b3,c3,d3]=fdr_bh(lpv,0.05,'dep');
[a4,b4,c4,d4]=fdr_bh(lpvp,0.05,'dep');

figure;
subplot(2,2,1); imagesc(spv); caxis([0,0.05]); colorbar; axis square; title('Duration Cycle (FDR-adjusted p-val)')
subplot(2,2,3); imagesc(spvp); caxis([0,0.05]); colorbar; axis square; title('Period Cycle (FDR-adjusted p-val)')
subplot(2,2,2); imagesc(lpv); caxis([0,0.05]); colorbar; axis square; title('Duration Linear (FDR-adjusted p-val)')
subplot(2,2,4); imagesc(lpvp); caxis([0,0.05]); colorbar; axis square; title('Period Linear (FDR-adjusted p-val)')



%%


%%

% [Linear fit reconstructed, sine reconstructed, ones], predictors
X=[linBins sineBins ones(size(linBins))];
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

%%
clear linFitDurLeft sinFitDurLeft linFitDurRight sinFitDurRight linFitDurBoth sinFitDurBoth
clear linFitPerLeft sinFitPerLeft linFitPerRight sinFitPerRight linFitPerBoth sinFitPerBoth
for ns = whichSubs
    for nm = mapList
        for nh = 1:length(hemispheres)
            
            %Duration
            % Left
            ind = (nm-1)*length(hemispheres)+nh;
            linFitDur(ns,nm,nh,:) = stats{ns}.linFitDur(ind,:);
            sinFitDur(ns,nm,nh,:) = stats{ns}.sinFitDur(ind,:);
            
            %Right
            %    linFitDurRight(ns,nm,nh,:) = stats{ns}.linFitDur(ind,:);
            %    sinFitDurRight(ns,nm,nh,:) = stats{ns}.sinFitDur(ind,:);
            
            %Both
            %    linFitDurBoth(ns,:) = [stats{ns}.linFitDur(1,:),stats{ns}.linFitDur(2,:)];
            %    sinFitDurBoth(ns,:) = [stats{ns}.sinFitDur(1,:),stats{ns}.sinFitDur(2,:)];
            
            %Period
            %Left
            linFitPer(ns,nm,nh,:) = stats{ns}.linFitPer(ind,:);
            sinFitPer(ns,nm,nh,:) = stats{ns}.sinFitPer(ind,:);
            
            %Right
            %    linFitPerRight(ns,nm,nh,:) = stats{ns}.linFitPer(ind,:);
            %    sinFitPerRight(ns,nm,nh,:) = stats{ns}.sinFitPer(ind,:);
            
            %Both
            %   linFitPerBoth(ns,:) = [stats{ns}.linFitPer(1,:),stats{ns}.linFitPer(2,:)];
            %   sinFitPerBoth(ns,:) = [stats{ns}.sinFitPer(1,:),stats{ns}.sinFitPer(2,:)];
        end
    end
end
%%
%Duration
% [lindurH, lindurP, lindurCI, lindurStats] = ttest(linFitDurBoth);
% [sindurH, sindurP, sindurCI, sindurStats] = ttest(sinFitDurBoth);

% Duration left vs right
[~,lindiffdurP, lindiffdurCI, lindiffdurStats]=ttest(linFitDurLeft,linFitDurRight);
[~,sindiffdurP, sindiffdurCI, sindiffdurStats]=ttest(sinFitDurLeft,sinFitDurRight);

%Period
% [linperH, linperP, linperCI, linperStats] = ttest(linFitPerBoth);
% [sinperH, sinperP, sinperCI, sinperStats] = ttest(sinFitPerBoth);

[~,lindiffperP, lindiffperCI, lindiffperStats]=ttest(linFitPerLeft,linFitPerRight);
[~,sindiffperP, sindiffperCI, sindiffperStats]=ttest(sinFitPerLeft,sinFitPerRight);

%% remake ANOVA structure
VEs=nan([length(subjectOrder) length(mapNames) length(hemispheres)]);

% sub, map, hemi

Constant=VEs;
Slope=VEs;
Phase=VEs;
Frequency=VEs;
Scale=VEs;



% veIndices=data.ves>0.1 & data.x0s>0.05 & data.x0s<1 ;
% VEs(n,m,whichHemi)=mean(data.ves(veIndices));

for subs = 1:length(subjectOrder)
    for maps = 1:length(mapNames)
        for hemis = 1:length(hemispheres)
            
            %                 Constant(n,m,1)=linFitPer(hemis);
            Constant(subs,maps,hemis)=linFitDur(subs,maps,hemis,2);
            Slope(subs,maps,hemis) = linFitDur(subs,maps,hemis,1);
            Frequency(subs,maps,hemis)=sinFitDur(subs,maps,hemis,1);
            Phase(subs,maps,hemis)=sinFitDur(subs,maps,hemis,2);
            Scale(subs,maps,hemis)=sinFitDur(subs,maps,hemis,3);
            
            Constantp(subs,maps,hemis)=linFitPer(subs,maps,hemis,2);
            Slopep(subs,maps,hemis) = linFitPer(subs,maps,hemis,1);
            Frequencyp(subs,maps,hemis)=sinFitPer(subs,maps,hemis,1);
            Phasep(subs,maps,hemis)=sinFitPer(subs,maps,hemis,2);
            Scalep(subs,maps,hemis)=sinFitPer(subs,maps,hemis,3);
        end
    end
end
%%
mapNames = mapNames;
subjectOrder = subjectOrder;

hemisphereGroups=cat(3, ones(size(VEs(:,:,1))), ones(size(VEs(:,:,1)))*2);
tmp=1:length(subjectOrder);
subjectLabels=cat(3, repmat(tmp(:), [1,length(mapNames)]), repmat(tmp(:), [1,length(mapNames)]));
mapLabels=cat(3, repmat(1:length(mapNames), [length(subjectOrder),1]), repmat(1:length(mapNames), [length(subjectOrder),1]));
mapLabels=mapLabels(:);
subjectLabels=subjectLabels(:);
hemisphereLabels=hemisphereGroups(:);
subjectLabels=subjectOrder(subjectLabels);
mapLabels=mapNames(mapLabels);
hemiTmp={'L', 'R'};
hemisphereLabels=hemiTmp(hemisphereLabels);

%% Constant component of the linear fit

[p, tmp, statsOut] = anovan(Constant(:),{subjectLabels  mapLabels hemisphereLabels}, 'varnames', {'subject', 'map', 'hemi'});%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [1 2])

% without hemi
[p, tmp, statsOut] = anovan(Constant(:),{subjectLabels  mapLabels}, 'varnames', {'subject', 'map'});%Multiple comparison tests on ANOVA output
%% Slope component of the linear fit

[p, tmp, statsOut] = anovan(Slope(:),{subjectLabels  mapLabels hemisphereLabels}, 'varnames', {'subject', 'map', 'hemi'});%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [2])

% without hemi
[p, tmp, statsOut] = anovan(Slope(:),{mapLabels}, 'varnames', {'map'});%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [1])
%% Frequency component of the cyclical fit

[p, tmp, statsOut] = anovan(Frequency(:),{subjectLabels  mapLabels hemisphereLabels}, 'varnames', {'subject', 'map', 'hemi'});%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [2])

% without hemi
[p, tmp, statsOut] = anovan(Frequency(:),{subjectLabels  mapLabels}, 'varnames', {'subject', 'map'});%Multiple comparison tests on ANOVA output

%% Phase component of the cyclical fit

[p, tmp, statsOut] = anovan(Phase(:),{subjectLabels  mapLabels hemisphereLabels}, 'varnames', {'subject', 'map', 'hemi'});%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [2])

% without hemi
[p, tmp, statsOut] = anovan(Phase(:),{subjectLabels  mapLabels}, 'varnames', {'subject', 'map'});%Multiple comparison tests on ANOVA output

%% Scale component of the cyclical fit

[p, tmp, statsOut] = anovan(Scale(:),{subjectLabels  mapLabels hemisphereLabels}, 'varnames', {'subject', 'map', 'hemi'});%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [2])

% without hemi
[p, tmp, statsOut] = anovan(Scale(:),{subjectLabels  mapLabels}, 'varnames', {'subject', 'map'});%Multiple comparison tests on ANOVA output

%% Constant component of the linear fit

[p, tmp, statsOut] = anovan(Constantp(:),{subjectLabels  mapLabels hemisphereLabels}, 'varnames', {'subject', 'map', 'hemi'});%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [2])

% without hemi % sub
[p, tmp, statsOut] = anovan(Constantp(:),{subjectLabels  mapLabels}, 'varnames', {'subject', 'map'});%Multiple comparison tests on ANOVA output
%% Slope component of the linear fit

[p, tmp, statsOut] = anovan(Slopep(:),{subjectLabels  mapLabels hemisphereLabels}, 'varnames', {'subject', 'map', 'hemi'});%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [2])

% without hemi
[p, tmp, statsOut] = anovan(Slopep(:),{mapLabels}, 'varnames', {'map'});%Multiple comparison tests on ANOVA output
% figure; results=multcompare(statsOut, 'Dimension', [1])
%% Frequency component of the cyclical fit

[p, tmp, statsOut] = anovan(Frequencyp(:),{subjectLabels  mapLabels hemisphereLabels}, 'varnames', {'subject', 'map', 'hemi'});%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [2])

% without hemi
[p, tmp, statsOut] = anovan(Frequencyp(:),{subjectLabels  mapLabels}, 'varnames', {'subject', 'map'});%Multiple comparison tests on ANOVA output

%% Phase component of the cyclical fit

[p, tmp, statsOut] = anovan(Phasep(:),{subjectLabels  mapLabels hemisphereLabels}, 'varnames', {'subject', 'map', 'hemi'});%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [2 3])

% without hemi
[p, tmp, statsOut] = anovan(Phasep(:),{subjectLabels  mapLabels}, 'varnames', {'subject', 'map'});%Multiple comparison tests on ANOVA output

%% Scale component of the cyclical fit

[p, tmp, statsOut] = anovan(Scalep(:),{subjectLabels  mapLabels hemisphereLabels}, 'varnames', {'subject', 'map', 'hemi'});%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [2])

[p, tmp, statsOut] = anovan(Scalep(:),{subjectLabels  mapLabels}, 'varnames', {'subject', 'map'});%Multiple comparison tests on ANOVA output



% figure; imagesc([lindurP;linperP],[0,0.1]); colorbar;
%%

for thisSub = whichSub
    stats{thisSub}=PlotRoiDistanceAllRoiDurationPeriod(plotdata, 0.1, [0 1 0 0 0 1 1],1); %Plot everything [1 1 1 0 1 1 1] %Tuning widths only [0 0 0 0 0 1 1]
end


%%

subsTmp=whichSubs;
for whichSubs=subsTmp
    for map=mapList
        index=(map-1)*2+[1:2];
        stats{whichSubs}.pXprog(index,:)=statsTmp{whichSubs}.pXprog(index,:);
        stats{whichSubs}.pYprog(index,:)=statsTmp{whichSubs}.pYprog(index,:);
        stats{whichSubs}.rs(index)=statsTmp{whichSubs}.rs(index);
        stats{whichSubs}.ns(index)=statsTmp{whichSubs}.ns(index);
        stats{whichSubs}.rsxOddEven(index)=statsTmp{whichSubs}.rsxOddEven(index);
        stats{whichSubs}.rsyOddEven(index)=statsTmp{whichSubs}.rsyOddEven(index);
        stats{whichSubs}.nsOddEven(index)=statsTmp{whichSubs}.nsOddEven(index);
        try
            stats{whichSubs}.pSMajProg(index,:)=statsTmp{whichSubs}.pSMajProg(index,:);
            stats{whichSubs}.pSMinProg(index,:)=statsTmp{whichSubs}.pSMinProg(index,:);
        catch
            stats{whichSubs}=rmfield(stats{whichSubs}, 'pSMajProg');
            stats{whichSubs}=rmfield(stats{whichSubs}, 'pSMinProg');
            stats{whichSubs}.pSMajProg(index,:)=statsTmp{whichSubs}.pSMajProg(index,:);
            stats{whichSubs}.pSMinProg(index,:)=statsTmp{whichSubs}.pSMinProg(index,:);
        end
        stats{whichSubs}.tuningSlopesMajor(index,:)=statsTmp{whichSubs}.tuningSlopesMajor(index,:);
        stats{whichSubs}.tuningSlopesMinor(index,:)=statsTmp{whichSubs}.tuningSlopesMinor(index,:);
        stats{whichSubs}.tuningXs(index,:)=statsTmp{whichSubs}.tuningXs(index,:);
        stats{whichSubs}.tuningYmajor(index,:)=statsTmp{whichSubs}.tuningYmajor(index,:);
        stats{whichSubs}.tuningYminor(index,:)=statsTmp{whichSubs}.tuningYminor(index,:);
    end
end
whichSubs=subsTmp;
%%
whichmaps=[3 9 10 11];
clear plotdatatmp
for n=1:length(whichmaps)
    for m=1:3
        plotdatatmp{n,m}=plotdata{whichmaps(n), m};
    end
end

%%
figure;

subplot(1,2,1)
hold on;
for n = 1:6
    for m = 1
        plot([stats{n}.linFitDur(m,1),stats{n}.linFitDur(m+1,1)])
    end
end

subplot(1,2,2)
hold on;
for n = 1:6
    for m = 1
        plot([stats{n}.linFitDur(m,2),stats{n}.linFitDur(m+1,2)])
    end
end

%%
figure;

subplot(2,2,1)
hold on;
for n = 1:6
    for m = 1
        plot([stats{n}.sinFitDur(m,1),stats{n}.sinFitDur(m+1,1)])
    end
end


subplot(2,2,2)
hold on;
for n = 1:6
    for m = 1
        plot([stats{n}.sinFitDur(m,2),stats{n}.sinFitDur(m+1,2)])
    end
end

subplot(2,2,3)
hold on;
for n = 1:6
    for m = 1
        plot([stats{n}.sinFitDur(m,3),stats{n}.sinFitDur(m+1,3)])
    end
end

subplot(2,2,4)
hold on;
for n = 1:6
    for m = 1
        plot([stats{n}.sinFitDur(m,4),stats{n}.sinFitDur(m+1,4)])
    end
end

%% Tuning width versus duration
tmp1 = (dataAllSub.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves >= 0.05...
    & dataAllSub.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05...
    & dataAllSub.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 2 ...
    & dataAllSub.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s >= 0.05 ...
    & dataAllSub.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s <= 2);

tmp2 = (dataAllSub.AnteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves >= 0.05...
    & dataAllSub.AnteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05...
    & dataAllSub.AnteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 2 ...
    & dataAllSub.AnteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s >= 0.05 ...
    & dataAllSub.AnteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s <= 2);

tmp3 = (dataAllSub.MiddleBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves >= 0.05...
    & dataAllSub.MiddleBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05...
    & dataAllSub.MiddleBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 2 ...
    & dataAllSub.MiddleBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s >= 0.05 ...
    & dataAllSub.MiddleBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s <= 2);

tmp4 = (dataAllSub.MiddleBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves >= 0.05...
    & dataAllSub.MiddleBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05...
    & dataAllSub.MiddleBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 2 ...
    & dataAllSub.MiddleBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s >= 0.05 ...
    & dataAllSub.MiddleBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s <= 2);

tmp5 = (dataAllSub.PosteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves >= 0.05...
    & dataAllSub.PosteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05...
    & dataAllSub.PosteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 2 ...
    & dataAllSub.PosteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s >= 0.05 ...
    & dataAllSub.PosteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s <= 2);

tmp6 = (dataAllSub.PosteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves >= 0.05...
    & dataAllSub.PosteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05...
    & dataAllSub.PosteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 2 ...
    & dataAllSub.PosteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s >= 0.05 ...
    & dataAllSub.PosteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s <= 2);

tmp7 = (dataAllSub.Premotor.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves >= 0.05...
    & dataAllSub.Premotor.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05...
    & dataAllSub.Premotor.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 2 ...
    & dataAllSub.Premotor.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s >= 0.05 ...
    & dataAllSub.Premotor.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s <= 2);

tmp8 = (dataAllSub.Premotor.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves >= 0.05...
    & dataAllSub.Premotor.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s >= 0.05...
    & dataAllSub.Premotor.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s <= 2 ...
    & dataAllSub.Premotor.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s >= 0.05 ...
    & dataAllSub.Premotor.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.y0s <= 2);


ATL = corrcoef(dataAllSub.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp1),dataAllSub.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.sigmas(tmp1));
ATR = corrcoef(dataAllSub.AnteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp2),dataAllSub.AnteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.sigmas(tmp2));
MTL = corrcoef(dataAllSub.MiddleBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp3),dataAllSub.MiddleBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.sigmas(tmp3));
MTR = corrcoef(dataAllSub.MiddleBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp4),dataAllSub.MiddleBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.sigmas(tmp4));
PTL = corrcoef(dataAllSub.PosteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp5),dataAllSub.PosteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.sigmas(tmp5));
PTR = corrcoef(dataAllSub.PosteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp6),dataAllSub.PosteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.sigmas(tmp6));
PML = corrcoef(dataAllSub.Premotor.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp7),dataAllSub.Premotor.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.sigmas(tmp7));
PMR = corrcoef(dataAllSub.Premotor.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp8),dataAllSub.Premotor.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.sigmas(tmp8));

ATLp = r2p(ATL(1,2), sqrt(length(dataAllSub.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp1))));
ATRp = r2p(ATR(1,2), sqrt(length(dataAllSub.AnteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp2))));
MTLp = r2p(MTL(1,2), sqrt(length(dataAllSub.MiddleBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp3))));
MTRp = r2p(MTR(1,2), sqrt(length(dataAllSub.MiddleBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp4))));
PTLp = r2p(PTL(1,2), sqrt(length(dataAllSub.PosteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp5))));
PTRp = r2p(PTR(1,2), sqrt(length(dataAllSub.PosteriorBelt.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp6))));
PMLp = r2p(PML(1,2), sqrt(length(dataAllSub.Premotor.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp7))));
PMRp = r2p(PMR(1,2), sqrt(length(dataAllSub.Premotor.Right.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.x0s(tmp8))));

%%

% tmp = fieldnames(dataS2.AnteriorBelt.Left);
figure; hold on; for n =1:length(tmp); a= unique(dataS2.AnteriorBelt.Left.(tmp{n}).All.exp); plot(a); end

[a,b,c,d] = ttest(dataS2.AnteriorBelt.Left.Tonotopy.All.ves, dataS2.AnteriorBelt.Left.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.ves)



%%


for n = whichSubs
    for m = mapList
%         for q = 1:length(hemispheres)
            
    veIndices = dataS1{n,m}.x0s >= 0.05 & dataAll{n,m}.x0s <= 1 & dataAll{n,m}.y0s >= 0.05 & dataAll{n,m}.y0s <=1 & dataAll{n,m}.ves > 0.1;
    figure;     
    plot(dataAll{n,m}.x0s(veIndices),dataAll{n,m}.sigmas(veIndices), '.')
    end
end


%% 
for n = 1 : length(subjectOrder)
    for m = 1:length(mapNames)
        for lh = 1:2%length(hemispheres)
            
            
            data(n,m,lh)=length(eval(strcat(subjectOrder{n}, '.', mapNames{m}, '.', hemispheres{lh}, '.Lin2dOvalGaussianDurationPeriodexpIntensityfree.All.roiIndices')));
        end
    end
end
    
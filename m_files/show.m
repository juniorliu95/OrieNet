function h = show(im, param2, param3)

    Octave = exist('OCTAVE_VERSION', 'builtin') == 5; % Are we running under Octave

    s = warning('query','all');                 % Record existing warning state.
    warn_state = onCleanup (@() warning(s));    % Restore warnings at the end
    warning('off');                             % Turn off warnings that might arise if image
                                                % has to be rescaled to fit on screen

    % Check case where im is an image filename rather than image data
    if ~isnumeric(im) && ~islogical(im) 
        Title = im;            % Default title is file name
        im = imread(im);
    else
        Title = inputname(1);  % Default title is variable name of image data
    end

    figNo = -1;                % Default value indicating create new figure
    
    % If two arguments check type of 2nd argument to see if it is the title or
    % figure number that has been supplied 
    if nargin == 2
        if strcmp(class(param2),'char')
            Title = param2;
        elseif isnumeric(param2) && length(param2) == 1
            figNo = param2;
        else
            error('2nd argument must be a figure number or title');
        end
    elseif nargin == 3
        figNo = param2;
        Title = param3;

        if ~strcmp(class(Title),'char')
            error('Title must be a string');
        end
        
        if ~isnumeric(param2) || length(param2) ~= 1
            error('Figure number must be an integer');
        end
    end

    if figNo > 0           % We have a valid figure number
        figure(figNo);     % Reuse or create a figure window with this number
        subplot('position',[0 0 1 1]); % Use the whole window
    elseif figNo == -1
        figNo = figure;        % Create new figure window
        subplot('position',[0 0 1 1]); % Use the whole window
    end

    if ndims(im) == 2          % Display as greyscale
        imagesc(im);
        colormap(gray(256));   % Ensure we have a full 256 level greymap
    else
        figure;
        imshow(im);            % Display as RGB
    end

    if figNo == 0                 % Assume we are trying to do a subplot 
        figNo = gcf;              % Get the current figure number
        axis('image'); axis('off');
        title(Title);             % Use a title rather than rename the figure
    else
        axis('image'); axis('off');
        set(figNo,'name', ['  ' Title])
        if ~Octave
            truesize(figNo);
        end
    end

    if nargout == 1
       h = figNo;
    end

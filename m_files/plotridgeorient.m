% PLOTRIDGEORIENT - plot of ridge orientation data
%
% Usage:   plotridgeorient(orient, spacing, im, figno)
%
%        orientim - Ridge orientation image (obtained from RIDGEORIENT)
%        spacing  - Sub-sampling interval to be used in ploting the
%                   orientation data the (Plotting every point is
%                   typically not feasible) 
%        im       - Optional fingerprint image in which to overlay the
%                   orientation plot.
%        figno    - Optional figure number for plot
%
% A spacing of about 20 is recommended for a 500dpi fingerprint image
%
% See also: RIDGEORIENT, RIDGEFREQ, FREQEST, RIDGESEGMENT

% Peter Kovesi  
% School of Computer Science & Software Engineering
% The University of Western Australia
% pk at csse uwa edu au
% http://www.csse.uwa.edu.au/~pk
%
% January 2005

function s_orient = plotridgeorient(orient, spacing, im, figno, color)

    if fix(spacing) ~= spacing
	error('spacing must be an integer');
    end
    
    [rows, cols] = size(orient);
    
    lw = 3;             % linewidth
    len = 0.8*spacing;  % length of orientation lines

    % Subsample the orientation data according to the specified spacing

    s_orient = orient(spacing/2:spacing:rows-spacing/2, ...
	      spacing/2:spacing:cols-spacing/2);  %%做了下修改，变为10*10的方向场
    %  s_orient = orient(spacing:spacing:rows, ...
	%	      spacing:spacing:cols);

    xoff = len/2*cos(s_orient);
    yoff = len/2*sin(s_orient);    
    
%     if nargin >= 3     % Display fingerprint image
% 	if nargin == 4
	    show(im, figno); hold on
%     else
% 	    show(im); hold on
% 	end
%     end
    
    % Determine placement of orientation vectors
    [x,y] = meshgrid(spacing/2:spacing:cols-spacing/2, ...
		     spacing/2:spacing:rows-spacing/2);
    
    x = x-xoff;
    y = y-yoff;
    
    % Orientation vectors
    u = xoff*2;
    v = yoff*2;
    
    quiver(x,y,u,v,0,'.','linewidth',lw, 'color',color);
    
    axis equal, axis ij,  hold off
    

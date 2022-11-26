import torch
from PIL import ImageColor
import plotly.graph_objects as go
import plotly.colors


def get_color(colorscale_name, loc):
    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale_name)

    if hasattr(loc, "__iter__"):
        intermediate_colors = [get_continuous_color(colorscale, x) for x in loc]
        return intermediate_colors
    return get_continuous_color(colorscale, loc)

def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.
    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]
    Others are just swatches that need to be constructed into a colorscale:
        viridis_colors, D = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)
    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    intermediate_color = plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )
    return intermediate_color

def get_plotly_colors(num_points, colorscale):
    color_steps = torch.linspace(start=0, end=1, steps=num_points).tolist()
    colors = get_color(colorscale, color_steps)
    colors = [plotly.colors.unlabel_rgb(color) for color in colors]
    colors = torch.tensor(colors, dtype=torch.float, device='cuda').view(1, num_points, 3)
    colors = colors.div(255.0).add(-0.5).mul(2)  # Map [0, 255] RGB colors to [-1, 1]
    return colors  # (1, P, 3)

def get_colormap(flow, H, W):
    points = flow.permute(0, 2, 3, 1)
    points = points.reshape(points.size(0), points.size(1) * points.size(2), 2)  # (N, K*P, 2)
    num_points = points.size(1)
    colorscale = ['plasma']
    colors = torch.cat([get_plotly_colors(num_points, c) for c in colorscale], 1)  # (1, K*P, 3)
    colors = colors.reshape(H, W, 3).cpu().numpy() * 255 
    return colors
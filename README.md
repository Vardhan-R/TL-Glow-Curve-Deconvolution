# [TL Glow Curve Deconvolution](https://tlgcdeconv.streamlit.app)

Thermoluminescence glow curve deconvolution by minimising the figure of merit using numerical methods.

## Usage

1. Open [the Streamlit app](https://tlgcdeconv.streamlit.app).
2. Upload the file containing the data (in csv format only). The data should be as shown below: no headers, the first column must contain the temperature values (in K) and the second column contains the intensitie values (in a.u.).
![csv file example](images/csv_file_example.png)
3. Click the "Upload" button.
4. The program attempts to identify the number of peaks, locate the maxima of the data and display them.
5. The initial values of the parameters are determined by these maxmima (as they are computed using numerical methods) and displayed in the graph below.
6. One can change the parameters provided. **It is strongly recommended to set the initial $T_m$ and $I_m$ values close to the expected maxima of the individual peaks, as this would help with convergence.** See [below](#choosing-the-parameters-wisely) for more information on how to choose the parameters wisely.
6. Click the "Submit" button to update the parameters.
7. Click the "Execute" button to apply the chosen numerical method and determine the parameters. The fitted parameters, final FOM and plots are then displayed.

## Choosing the Parameters, Wisely

### Number of peaks

- If the fitted curve has an unexpectedly peaks where it is not supposed to, there may be more peaks which have not been initialised.

### $T_m$

- Set each $T_m$ to the expected value of each of the peaks.

### $I_m$

- If two peaks seem have their maxima close by, then initialise their corresponding $I_m$ values to be slightly lesser, such that the sum of the peaks would fit the data closely.

### $b$ and $E$

- The initial values don't matter much.
- Set $b$ initially around $1.6$.
- Set $E$ initially around $1$.

### Scale factor

- Choose the scale factor such that the intensities, when divided by the scale factor, lie roughly in the *tens to hundreds* range.

### Method

- During testing, it was observed that `SLSQP` works best, and `COBYQA` may not work.

## Example

See the example below.
![initial values](images/init_vals.png)
![fitted parameters](images/fitted_params.png)
![deconvoluted graph](images/deconv_graph.png)

## To Do

- Use the double derivative method to identify the peaks.
- Add an option to set the inital values of the base term parameters too.
- Add an option to export the fitted parameters.
- Better layout for the fitting part.

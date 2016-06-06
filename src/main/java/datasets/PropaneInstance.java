package datasets;

public class PropaneInstance implements Instance {
    private double[] frequencySpectrum;
    private double fuelLevel;

    public PropaneInstance(double[] frequencySpectrum, double fuelLevel) {
        this.frequencySpectrum = frequencySpectrum;
        this.fuelLevel = fuelLevel;
    }

    @Override
    public double[] getInput() {
        return frequencySpectrum;
    }

    @Override
    public double getOutput() {
        return fuelLevel;
    }

    @Override
    public double[] getPossibleOutputs() {
        return new double[] {-1, 1};
    }

    @Override
    public double getError(double y) {
        if (y * fuelLevel <= 0) {
            return 1;
        } else {
            return 0;
        }
    }
}

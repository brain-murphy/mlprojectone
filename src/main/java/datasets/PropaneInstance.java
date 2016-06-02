package datasets;

public class PropaneInstance implements Instance {
    private double[] frequencySpectrum;
    private double tankWeight;

    public PropaneInstance(double[] frequencySpectrum, double tankWeight) {
        this.frequencySpectrum = frequencySpectrum;
        this.tankWeight = tankWeight;
    }

    @Override
    public double[] getInput() {
        return frequencySpectrum;
    }

    @Override
    public double getOutput() {
        return tankWeight;
    }
}

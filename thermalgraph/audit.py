from .data import ResearchArchive


def main() -> None:
    archive = ResearchArchive()
    assert len(archive.nodes) == 72
    assert archive.frame_count == 181
    assert len(archive.topology["directed_edges"]) == 324
    assert len(archive.current_temperatures(0)) == 72
    assert len(archive.forecast(0, 60)) == 72
    probability = archive.risk_summary(60, 40, 60)["probability"]
    assert 0 <= probability <= 1
    print("Archive audit passed: 181 frames, 72 nodes, 324 candidate edges.")


if __name__ == "__main__":
    main()

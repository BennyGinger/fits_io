from fits_io.readers._types import StatusFlag


INFO_NAMESPACE = "fits_io"

class InfoProfile:
    """
    Class to build the string for ImageJ Info metadata
    Attributes:
        status [StatusFlag]: The processing status of the image, either 'active' or 'skip'.
        user_name [str]: The name of the user who processed the image. Default is "unknown". 
        namespace [str]: The namespace to use for the metadata keys. Default is "fits_io".  
    """
    __slots__ = ['_status', '_user', 'namespace']
    
    def __init__(self, status: StatusFlag, user: str = "unknown", namespace: str = INFO_NAMESPACE) -> None:
        self._status = status 
        self._user = user
        self.namespace = namespace    
    
    @property
    def status(self) -> str:
        return f"{self.namespace}.status: {self._status}"
    
    @property
    def user(self) -> str:
        return f"{self.namespace}.user: {self._user}"

    @property
    def export(self) -> str:
        lines = [
            self.status,
            self.user,
            ]
        return "\n".join(lines) + "\n"